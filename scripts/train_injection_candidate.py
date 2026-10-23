"""Train an isolated injection-head candidate on pinned bilingual data.

Selection uses only the source's train/validation splits. The source test and
cross-source corpora are diagnostics, never hyperparameter selection inputs.
"""

import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from trustlaya.calibration import fit_temperature, metrics as calibration_metrics
from trustlaya.model import BACKBONE, TrustLaya
from trustlaya.utils import device, seed_all
from train_advanced_pii import embeddings


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "MelikeErdogan/bpi-guard-dataset"
REVISION = "a029afd7b0ed3979d1a3446b0e7c54870ff1a63d"
DEST = ROOT / "models/candidates/injection_bpi"


def key(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def load():
    splits, hashes = {}, {}
    for split in ("train", "validation", "test"):
        path = Path(hf_hub_download(SOURCE, f"{split}.jsonl", repo_type="dataset",
                                    revision=REVISION))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.open()]
        assert all(row["label"] in ("benign", "injection") for row in rows)
        splits[split] = [{"text": row["text"], "label": int(row["label"] == "injection"),
                          "lang": row.get("lang"), "source": row.get("source"),
                          "original_text": row.get("original_text")}
                         for row in rows if row.get("text")]
    train_keys = {key(row["text"]) for row in splits["train"]}
    validation_keys = {key(row["text"]) for row in splits["validation"]}
    # Remove direct text and original-source overlap from the published test.
    excluded = Counter()
    test = []
    for row in splits["test"]:
        if key(row["text"]) in train_keys | validation_keys:
            excluded["exact_text_overlap"] += 1
        elif row["original_text"] and key(row["original_text"]) in train_keys | validation_keys:
            excluded["original_text_overlap"] += 1
        else:
            test.append(row)
    splits["test"] = test
    return splits, hashes, dict(excluded)


def measures(rows, logits, temperature, threshold):
    truth = np.array([row["label"] for row in rows])
    probability = expit(logits / temperature)
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {"n": len(rows), "positive": int(truth.sum()), "threshold": float(threshold),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "false_positive_rate": float(fp / max(1, tn + fp)),
            "false_negative_rate": float(fn / max(1, fn + tp)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "calibration": calibration_metrics(truth, probability)}


def main():
    seed_all(42)
    torch.set_num_threads(4)
    split, hashes, excluded = load()
    base = ROOT / "models/trustlaya-s-v2"
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(base / "model.safetensors")
    model.to(device()).eval()
    vectors = {name: embeddings(model, tokenizer, rows) for name, rows in split.items()}
    truth = {name: np.array([row["label"] for row in rows]) for name, rows in split.items()}
    base_temperature = json.loads((base / "calibration.json").read_text())["prompt_injection"]
    weights = model.heads.risks.weight[2].detach().cpu().numpy().copy()
    bias = model.heads.risks.bias[2].detach().cpu().numpy().copy()
    baseline = {name: measures(rows, vectors[name] @ weights + bias, base_temperature, .5)
                for name, rows in split.items() if name != "train"}
    eligible = []
    trials = []
    for c in (.001, .01, .1, 1.0):
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42)
        clf.fit(vectors["train"], truth["train"])
        dev_logits = clf.decision_function(vectors["validation"])
        temperature = fit_temperature(truth["validation"], dev_logits)
        for threshold in np.arange(.1, .951, .05):
            score = measures(split["validation"], dev_logits, temperature, threshold)
            trial = (score["f1"], -score["false_positive_rate"], c, threshold,
                     clf, temperature, score)
            trials.append(trial)
            if score["false_positive_rate"] <= .10:
                eligible.append(trial)
    best = max(eligible or trials, key=lambda row: row[:2])
    _, _, chosen_c, threshold, clf, temperature, selected_validation = best
    candidate = {name: measures(rows, clf.decision_function(vectors[name]),
                                temperature, threshold)
                 for name, rows in split.items() if name != "train"}
    report = {"source": SOURCE, "source_revision": REVISION, "license": "MIT",
              "source_sha256": hashes, "rows": {name: len(rows) for name, rows in split.items()},
              "excluded_test_overlap": excluded,
              "test_warning": "Source test includes transformed/paraphrased examples; family overlap beyond explicit original_text is possible. The label includes jailbreak and social-engineering content as well as prompt injection. Do not compare directly to agentic tool-output results.",
              "selected_on": "validation only, maximum F1 subject to FPR <= 0.10 if feasible",
              "selected_C": chosen_c, "temperature": temperature,
              "threshold": float(threshold), "baseline_v2_at_0_5": baseline,
              "candidate": candidate,
              "selection_note": "No hyperparameter was chosen on test or cross-source corpora. Candidate is not promoted automatically."}
    DEST.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.heads.risks.weight[2].copy_(torch.tensor(clf.coef_[0], device=device(), dtype=torch.float32))
        model.heads.risks.bias[2].copy_(torch.tensor(clf.intercept_[0], device=device(), dtype=torch.float32))
    model.save(DEST / "model.safetensors")
    for file in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        shutil.copy2(base / file, DEST / file)
    calibration = json.loads((base / "calibration.json").read_text())
    calibration["prompt_injection"] = temperature
    (DEST / "calibration.json").write_text(json.dumps(calibration, indent=2))
    thresholds = json.loads((base / "decision_thresholds.json").read_text())
    thresholds["prompt_injection"] = float(threshold)
    (DEST / "decision_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    shutil.copy2(base / "policy.yaml", DEST / "policy.yaml")
    target = ROOT / "reports/injection_candidate_bpi.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_C": chosen_c, "threshold": threshold,
                      "baseline_test": baseline["test"],
                      "candidate_test": candidate["test"]}, indent=2))


if __name__ == "__main__":
    main()
