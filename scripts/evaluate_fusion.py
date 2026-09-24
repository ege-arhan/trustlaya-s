"""Choose a transparent fusion rule on the original validation split only."""

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from trustlaya.dataset import read_rows
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = read_rows(ROOT / "data/splits/val.jsonl")
    analyzer = Analyzer("onnx")
    outputs = [analyzer.analyze(row["text"]) for row in rows]
    truth = [int(any(row["labels"][task] for task in TASKS)) for row in rows]
    report = {"dataset": "original synthetic validation, English-labeled only",
              "target": "any of nine task labels positive", "threshold": 0.7,
              "methods": {}}
    for method in ("weighted", "interaction", "rule_constrained"):
        scores = [item["risk_fusion"]["alternatives"][method] for item in outputs]
        candidates = []
        for threshold in (i / 100 for i in range(5, 96, 5)):
            pred = [int(score >= threshold) for score in scores]
            tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
            candidates.append({
                "threshold": threshold,
                "f1": float(f1_score(truth, pred, zero_division=0)),
                "precision": float(precision_score(truth, pred, zero_division=0)),
                "recall": float(recall_score(truth, pred, zero_division=0)),
                "false_positive_rate": float(fp / max(1, fp + tn)),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })
        report["methods"][method] = {
            "at_0_70": next(row for row in candidates if row["threshold"] == 0.7),
            "best_validation_f1": max(candidates, key=lambda row: (row["f1"], -row["false_positive_rate"])),
            "best_validation_f1_at_fpr_le_0_10": max(
                (row for row in candidates if row["false_positive_rate"] <= 0.10),
                key=lambda row: (row["f1"], -row["false_positive_rate"]), default=None),
        }
    (ROOT / "reports/fusion_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
