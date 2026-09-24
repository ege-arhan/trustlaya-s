"""Train a versioned PII-head candidate on MIT Turkish data and synthetic anchors."""

import json
import random
import shutil
from pathlib import Path
import yaml

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoTokenizer

from trustlaya.calibration import fit_temperature, metrics
from trustlaya.dataset import read_rows
from trustlaya.evidence import extract, PII_TYPES
from trustlaya.model import TrustLaya, BACKBONE
from trustlaya.utils import device, normalize, seed_all

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "yusuf-said/turkish-privacy-filter-dataset"
PII_KEYS = {"account_number", "private_person", "private_phone", "private_email", "private_address"}
DEV_SCENARIOS = {"json_log", "chat_transcript"}
TEST_SCENARIOS = {"ocr_scan", "server_log", "call_center_log"}


def data():
    path = hf_hub_download(SOURCE, filename="tr_privacy_tr_curated.jsonl", repo_type="dataset")
    raw = [json.loads(line) for line in open(path)]
    rows = [{"text": row["text"], "label": int(any(
        key.split(":")[0] in PII_KEYS for key in row["spans"])),
        "scenario": row["info"]["scenario"]} for row in raw]
    split = {"train": [], "dev": [], "test": []}
    for row in rows:
        scenario = row["scenario"]
        name = "test" if scenario in TEST_SCENARIOS else "dev" if scenario in DEV_SCENARIOS else "train"
        split[name].append(row)
    assert set(r["scenario"] for r in split["train"]).isdisjoint(
        set(r["scenario"] for r in split["test"]))
    anchors = read_rows(ROOT / "data/splits/train.jsonl")
    rng = random.Random(42)
    anchors = rng.sample(anchors, 1000)
    split["train"].extend({"text": row["text"], "label": row["labels"]["pii"],
                           "scenario": "v1_synthetic_anchor"} for row in anchors)
    return split


def embeddings(model, tokenizer, rows, batch_size=64):
    result = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        tokens = tokenizer([normalize(row["text"]) for row in chunk],
                           padding="max_length", truncation=True, max_length=96,
                           return_tensors="pt")
        ids = tokens["input_ids"].to(device())
        mask = tokens["attention_mask"].to(device())
        with torch.inference_mode():
            hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        result.append(pooled.cpu().numpy())
    return np.concatenate(result)


def evaluate(logits, rows, temperature=1.0, threshold=.5):
    from scipy.special import expit
    labels = np.array([row["label"] for row in rows])
    probs = expit(logits / temperature)
    evidence = np.array([any(item["type"] in PII_TYPES for item in extract(row["text"]))
                         for row in rows], dtype=bool)
    pred = (probs >= threshold) | evidence
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"n": len(rows), "positive": int(labels.sum()),
            "threshold": threshold,
            "model_f1": float(f1_score(labels, probs >= threshold, zero_division=0)),
            "hybrid_f1": float(f1_score(labels, pred, zero_division=0)),
            "recall": float(tp / max(1, tp + fn)),
            "false_positive_rate": float(fp / max(1, fp + tn)),
            "calibration": metrics(labels, probs)}


def select_threshold(logits, rows, temperature):
    results = [evaluate(logits, rows, temperature, threshold=i / 100)
               for i in range(5, 100, 5)]
    eligible = [row for row in results if row["false_positive_rate"] <= .10]
    pool = eligible or results
    return max(pool, key=lambda row: (row["hybrid_f1"], -row["false_positive_rate"]))


def main():
    seed_all(42)
    torch.set_num_threads(4)
    split = data()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v1")
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(ROOT / "models/trustlaya-s-v1/model.safetensors")
    model.to(device()).eval()
    vectors = {name: embeddings(model, tokenizer, rows) for name, rows in split.items()}
    weights = model.heads.risks.weight[0].detach().cpu().numpy().copy()
    bias = model.heads.risks.bias[0].detach().cpu().numpy().copy()
    base_temperature = json.loads((ROOT / "models/trustlaya-s-v1/calibration.json").read_text())["pii"]
    base_dev_logits = vectors["dev"] @ weights + bias
    base_threshold = select_threshold(base_dev_logits, split["dev"], base_temperature)["threshold"]
    baseline = {name: evaluate(vectors[name] @ weights + bias, rows, base_temperature)
                for name, rows in split.items() if name != "train"}
    baseline_tuned = {name: evaluate(vectors[name] @ weights + bias, rows,
                                     base_temperature, base_threshold)
                      for name, rows in split.items() if name != "train"}
    train_y = np.array([row["label"] for row in split["train"]])
    dev_y = np.array([row["label"] for row in split["dev"]])
    candidates = []
    for c in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(C=c, max_iter=1000, random_state=42)
        clf.fit(vectors["train"], train_y)
        logits = clf.decision_function(vectors["dev"])
        temp = fit_temperature(dev_y, logits)
        score = select_threshold(logits, split["dev"], temp)
        candidates.append((score["hybrid_f1"] if score["false_positive_rate"] <= .10 else -1,
                           -score["false_positive_rate"], c, clf, temp, score))
    _, _, chosen_c, chosen, temperature, chosen_dev = max(candidates, key=lambda row: row[:2])
    threshold = chosen_dev["threshold"]
    candidate = {
        name: evaluate(chosen.decision_function(vectors[name]), rows, temperature)
        for name, rows in split.items() if name != "train"
    }
    candidate_tuned = {
        name: evaluate(chosen.decision_function(vectors[name]), rows, temperature, threshold)
        for name, rows in split.items() if name != "train"
    }
    report = {
        "source": SOURCE, "license": "MIT", "source_label": "synthetic_curated_not_human_verified",
        "split_by_scenario": {name: {"rows": len(rows),
                                     "scenarios": sorted({r["scenario"] for r in rows})}
                              for name, rows in split.items()},
        "anchor_rows": 1000, "selected_C": chosen_c, "pii_temperature": temperature,
        "baseline_v1_at_0_5": baseline, "candidate_v2_at_0_5": candidate,
        "baseline_v1_dev_tuned": baseline_tuned, "candidate_v2_dev_tuned": candidate_tuned,
        "selection_note": "C and thresholds selected on scenario-disjoint dev with FPR <= 0.10 when possible. Test used for a promotion check; subsequent selected-model test claims are selection-biased.",
    }
    dest = ROOT / "models/trustlaya-s-v2"
    dest.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.heads.risks.weight[0].copy_(torch.tensor(chosen.coef_[0], device=device(), dtype=torch.float32))
        model.heads.risks.bias[0].copy_(torch.tensor(chosen.intercept_[0], device=device(), dtype=torch.float32))
    model.save(dest / "model.safetensors")
    for file in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        shutil.copy2(ROOT / "models/trustlaya-s-v1" / file, dest / file)
    calibration = json.loads((ROOT / "models/trustlaya-s-v1/calibration.json").read_text())
    calibration["pii"] = temperature
    (dest / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (dest / "decision_thresholds.json").write_text(json.dumps({"pii": threshold}, indent=2))
    policy = yaml.safe_load((ROOT / "configs/policy.yaml").read_text())
    policy["pii"] = threshold
    (dest / "policy.yaml").write_text(yaml.safe_dump(policy, sort_keys=False))
    (ROOT / "reports/advanced_pii_experiment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"baseline_v1_dev_tuned": baseline_tuned,
                      "candidate_v2_dev_tuned": candidate_tuned,
                      "selected_C": chosen_c, "pii_temperature": temperature}, indent=2))


if __name__ == "__main__":
    main()
