"""Train a frozen-encoder injection head with bilingual and legacy anchors."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

from train_advanced_pii import embeddings
from train_injection_candidate import ROOT, load, measures
from trustlaya.calibration import fit_temperature
from trustlaya.dataset import read_rows
from trustlaya.model import BACKBONE, TrustLaya
from trustlaya.utils import device, seed_all


DEST = ROOT / "models/candidates/injection_joint"


def synthetic(split):
    return [{"text": row["text"], "label": row["labels"]["prompt_injection"]}
            for row in read_rows(ROOT / f"data/splits/{split}.jsonl")]


def main():
    seed_all(42)
    torch.set_num_threads(4)
    source, hashes, excluded = load()
    training = {"bpi": source["train"], "synthetic": synthetic("train")}
    validation = {"bpi": source["validation"], "synthetic": synthetic("val")}
    base = ROOT / "models/trustlaya-s-v2"
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(base / "model.safetensors")
    model.to(device()).eval()
    train_x = {name: embeddings(model, tokenizer, rows) for name, rows in training.items()}
    val_x = {name: embeddings(model, tokenizer, rows) for name, rows in validation.items()}
    train_y = {name: np.array([row["label"] for row in rows]) for name, rows in training.items()}
    val_y = {name: np.array([row["label"] for row in rows]) for name, rows in validation.items()}
    base_weight = model.heads.risks.weight[2].detach().cpu().numpy()
    base_bias = float(model.heads.risks.bias[2].detach().cpu())
    base_temp = json.loads((base / "calibration.json").read_text())["prompt_injection"]
    baseline = {name: measures(validation[name], val_x[name] @ base_weight + base_bias,
                               base_temp, .5) for name in validation}
    trials = []
    features = np.concatenate([train_x["bpi"], train_x["synthetic"]])
    labels = np.concatenate([train_y["bpi"], train_y["synthetic"]])
    dev_features = np.concatenate([val_x["bpi"], val_x["synthetic"]])
    dev_labels = np.concatenate([val_y["bpi"], val_y["synthetic"]])
    for anchor_weight in (1, 3, 8):
        sample_weights = np.concatenate([np.ones(len(training["bpi"])),
                                         np.full(len(training["synthetic"]), anchor_weight)])
        for c in (.01, .1, 1.0):
            classifier = LogisticRegression(C=c, max_iter=1000, random_state=42)
            classifier.fit(features, labels, sample_weight=sample_weights)
            temperature = fit_temperature(dev_labels, classifier.decision_function(dev_features))
            logits = {name: classifier.decision_function(val_x[name]) for name in validation}
            for threshold in np.arange(.3, .801, .05):
                scores = {name: measures(validation[name], logits[name], temperature, threshold)
                          for name in validation}
                trials.append({"anchor_weight": anchor_weight, "C": c,
                               "temperature": temperature, "threshold": float(threshold),
                               "bpi": scores["bpi"], "synthetic": scores["synthetic"],
                               "classifier": classifier})
    eligible = [row for row in trials if row["bpi"]["false_positive_rate"] <= .12
                and row["synthetic"]["false_positive_rate"] <=
                baseline["synthetic"]["false_positive_rate"] + .05]
    summary = {"source": "MelikeErdogan/bpi-guard-dataset + original synthetic training split",
               "bpi_revision": "a029afd7b0ed3979d1a3446b0e7c54870ff1a63d",
               "bpi_source_sha256": hashes,
               "train_rows": {name: len(rows) for name, rows in training.items()},
               "validation_rows": {name: len(rows) for name, rows in validation.items()},
               "baseline_development": baseline,
               "selection": "Mean BPI/synthetic development F1, BPI FPR <= 0.12, synthetic FPR <= baseline + 0.05. Test not used.",
               "eligible_trials": len(eligible),
               "trials": [{k: trial[k] for k in ("anchor_weight", "C", "temperature", "threshold")}
                          | {f"{name}_{metric}": trial[name][metric]
                             for name in ("bpi", "synthetic")
                             for metric in ("f1", "recall", "false_positive_rate")}
                          for trial in trials]}
    report_path = ROOT / "reports/injection_joint_selection.json"
    if not eligible:
        summary["status"] = "rejected_no_feasible_candidate"
        report_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"status": summary["status"], "eligible_trials": 0}))
        return
    selected = max(eligible, key=lambda row: ((row["bpi"]["f1"] + row["synthetic"]["f1"]) / 2,
                                             -row["bpi"]["false_positive_rate"]))
    clf = selected.pop("classifier")
    summary["status"] = "development_gate_passed"
    summary["selected"] = selected
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
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
    shutil.copy2(base / "policy.yaml", DEST / "policy.yaml")
    print(json.dumps({"status": summary["status"], "C": selected["C"],
                      "anchor_weight": selected["anchor_weight"],
                      "threshold": selected["threshold"],
                      "bpi_f1": selected["bpi"]["f1"],
                      "synthetic_f1": selected["synthetic"]["f1"],
                      "bpi_fpr": selected["bpi"]["false_positive_rate"],
                      "synthetic_fpr": selected["synthetic"]["false_positive_rate"]}, indent=2))


if __name__ == "__main__":
    main()
