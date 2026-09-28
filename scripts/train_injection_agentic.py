"""Train an isolated injection head on bilingual and paired agentic data."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
import yaml
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

from train_advanced_pii import embeddings
from train_injection_candidate import ROOT, load as load_bpi, key, measures
from trustlaya.calibration import fit_temperature
from trustlaya.dataset import read_rows
from trustlaya.model import BACKBONE, TrustLaya
from trustlaya.utils import device, seed_all


SOURCE = "3nesdeniz/agentic-prompt-injection-5k"
REVISION = "e658d94dcafc133e3a4483f9f9aa13507f3848c4"
DEST = ROOT / "models/candidates/injection_agentic"


def agentic(split):
    path = Path(hf_hub_download(SOURCE, f"data/{split}.parquet", repo_type="dataset",
                                revision=REVISION))
    raw = parquet.read_table(path).to_pylist()
    assert all(row["label"] in (0, 1) for row in raw)
    rows = [{"text": row["text"], "label": int(row["label"])} for row in raw]
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def legacy(split):
    return [{"text": row["text"], "label": row["labels"]["prompt_injection"]}
            for row in read_rows(ROOT / f"data/splits/{split}.jsonl")]


def main():
    seed_all(42)
    torch.set_num_threads(4)
    bpi, bpi_hashes, _ = load_bpi()
    agent_train, agent_train_hash = agentic("train")
    agent_val, agent_val_hash = agentic("validation")
    train = {"bpi": bpi["train"], "agentic": agent_train,
             "synthetic": legacy("train")}
    val = {"bpi": bpi["validation"], "agentic": agent_val,
           "synthetic": legacy("val")}
    train_keys = {key(row["text"]) for rows in train.values() for row in rows}
    excluded = {name: sum(key(row["text"]) in train_keys for row in rows)
                for name, rows in val.items()}
    val = {name: [row for row in rows if key(row["text"]) not in train_keys]
           for name, rows in val.items()}
    base = ROOT / "models/trustlaya-s-v2"
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(base / "model.safetensors")
    model.to(device()).eval()
    tokenizer = AutoTokenizer.from_pretrained(base)
    train_x = {name: embeddings(model, tokenizer, rows) for name, rows in train.items()}
    val_x = {name: embeddings(model, tokenizer, rows) for name, rows in val.items()}
    train_y = {name: np.array([row["label"] for row in rows]) for name, rows in train.items()}
    val_y = {name: np.array([row["label"] for row in rows]) for name, rows in val.items()}
    base_temp = json.loads((base / "calibration.json").read_text())["prompt_injection"]
    base_weight = model.heads.risks.weight[2].detach().cpu().numpy()
    base_bias = float(model.heads.risks.bias[2].detach().cpu())
    baseline = {name: measures(rows, val_x[name] @ base_weight + base_bias, base_temp, .5)
                for name, rows in val.items()}
    order = ("bpi", "agentic", "synthetic")
    train_features = np.concatenate([train_x[name] for name in order])
    train_labels = np.concatenate([train_y[name] for name in order])
    val_features = np.concatenate([val_x[name] for name in order])
    val_labels = np.concatenate([val_y[name] for name in order])
    trials = []
    for agent_weight in (1, 3, 6):
        sample_weights = np.concatenate([np.ones(len(train["bpi"])),
                                         np.full(len(train["agentic"]), agent_weight),
                                         np.ones(len(train["synthetic"]))])
        for c in (.01, .1, 1.0):
            clf = LogisticRegression(C=c, max_iter=1000, random_state=42)
            clf.fit(train_features, train_labels, sample_weight=sample_weights)
            temp = fit_temperature(val_labels, clf.decision_function(val_features))
            logits = {name: clf.decision_function(val_x[name]) for name in order}
            for threshold in np.arange(.35, .801, .05):
                score = {name: measures(val[name], logits[name], temp, threshold)
                         for name in order}
                trials.append({"agent_weight": agent_weight, "C": c, "temperature": temp,
                               "threshold": float(threshold), "scores": score,
                               "classifier": clf})
    eligible = [row for row in trials
                if row["scores"]["bpi"]["false_positive_rate"] <= .15
                and row["scores"]["agentic"]["false_positive_rate"] <= .15
                and row["scores"]["synthetic"]["false_positive_rate"] <= .05
                and row["scores"]["synthetic"]["f1"] >= .95]
    report = {"source": SOURCE, "revision": REVISION, "license": "CC-BY-4.0",
              "attribution": "Enes Deniz, Agentic Prompt-Injection 5K",
              "source_sha256": {"train": agent_train_hash, "validation": agent_val_hash},
              "bpi_sha256": bpi_hashes,
              "train_rows": {name: len(rows) for name, rows in train.items()},
              "validation_rows": {name: len(rows) for name, rows in val.items()},
              "validation_excluded_exact_training_overlap": excluded,
              "baseline_development": baseline,
              "selection": "Maximize minimum of BPI and agentic validation F1. Caps: BPI/agentic FPR <= 0.15, synthetic FPR <= 0.05 and F1 >= 0.95. No test used.",
              "eligible_trials": len(eligible),
              "trials": [{"agent_weight": row["agent_weight"], "C": row["C"],
                          "temperature": row["temperature"], "threshold": row["threshold"],
                          **{f"{name}_{metric}": row["scores"][name][metric]
                             for name in order for metric in ("f1", "recall", "false_positive_rate")}}
                         for row in trials]}
    report_path = ROOT / "reports/injection_agentic_selection.json"
    if not eligible:
        report["status"] = "rejected_no_feasible_candidate"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "eligible_trials": 0}))
        return
    selected = max(eligible, key=lambda row: (min(row["scores"]["bpi"]["f1"],
                                                   row["scores"]["agentic"]["f1"]),
                                             (row["scores"]["bpi"]["f1"] +
                                              row["scores"]["agentic"]["f1"]) / 2,
                                             -row["scores"]["agentic"]["false_positive_rate"]))
    clf = selected["classifier"]
    report["status"] = "development_gate_passed"
    report["selected"] = {k: selected[k] for k in ("agent_weight", "C", "temperature", "threshold", "scores")}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    DEST.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model.heads.risks.weight[2].copy_(torch.tensor(clf.coef_[0], device=device(), dtype=torch.float32))
        model.heads.risks.bias[2].copy_(torch.tensor(clf.intercept_[0], device=device(), dtype=torch.float32))
    model.save(DEST / "model.safetensors")
    for file in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        shutil.copy2(base / file, DEST / file)
    calibration = json.loads((base / "calibration.json").read_text())
    calibration["prompt_injection"] = selected["temperature"]
    (DEST / "calibration.json").write_text(json.dumps(calibration, indent=2))
    thresholds = json.loads((base / "decision_thresholds.json").read_text())
    thresholds["prompt_injection"] = selected["threshold"]
    (DEST / "decision_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    policy = yaml.safe_load((base / "policy.yaml").read_text())
    policy["prompt_injection"] = selected["threshold"]
    (DEST / "policy.yaml").write_text(yaml.safe_dump(policy, sort_keys=False))
    print(json.dumps({"status": report["status"], "selected": report["selected"]}, indent=2))


if __name__ == "__main__":
    main()
