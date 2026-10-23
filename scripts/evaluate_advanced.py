"""Evaluate a versioned candidate without changing v1 reports or weights."""

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from trustlaya.dataset import read_rows
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS

ROOT = Path(__file__).resolve().parents[1]


def main():
    candidate = ROOT / "models/trustlaya-s-v2"
    analyzer = Analyzer("torch_cpu", model_dir=candidate)
    thresholds = json.loads((candidate / "decision_thresholds.json").read_text())
    rows = read_rows(ROOT / "data/splits/test.jsonl")
    outputs = [analyzer.analyze(row["text"]) for row in rows]
    tasks = {}
    for task in TASKS:
        true = [row["labels"][task] for row in rows]
        threshold = thresholds.get(task, .5)
        pred = [int(out[task] >= threshold) for out in outputs]
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
        tasks[task] = {
            "threshold": threshold,
            "f1": float(f1_score(true, pred, zero_division=0)),
            "precision": float(precision_score(true, pred, zero_division=0)),
            "recall": float(recall_score(true, pred, zero_division=0)),
            "false_positive_rate": float(fp / max(1, fp + tn)),
            "false_negative_rate": float(fn / max(1, fn + tp)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }
    report = {"candidate": str(candidate), "dataset": "original 1975-row synthetic mixed-only test",
              "threshold_origin": "PII threshold selected on separate Turkish privacy dev scenarios",
              "mean_task_accuracy": sum((v["confusion_matrix"][0][0] + v["confusion_matrix"][1][1]) / len(rows)
                                        for v in tasks.values()) / len(tasks),
              "macro_f1": sum(v["f1"] for v in tasks.values()) / len(tasks),
              "tasks": tasks}
    confidences = [out["confidence"] for out in outputs]
    correct = [out["action"] == row["action"] for out, row in zip(outputs, rows)]
    report["confidence_distribution"] = {
        "min": min(confidences), "median": sorted(confidences)[len(confidences) // 2],
        "max": max(confidences),
        "bins": {f"{lo:.1f}-{lo + .1:.1f}": sum(lo <= score < lo + .1 if lo < .9 else lo <= score <= 1
                                                    for score in confidences)
                 for lo in (i / 10 for i in range(10))},
    }
    report["abstention_rate"] = sum(out["abstain"] for out in outputs) / len(outputs)
    report["selective_prediction"] = {}
    for threshold in (.5, .6, .7, .8, .9):
        selected = [i for i, score in enumerate(confidences) if score >= threshold]
        report["selective_prediction"][str(threshold)] = {
            "coverage": len(selected) / len(rows),
            "action_error_on_covered": (1 - sum(correct[i] for i in selected) / len(selected)) if selected else None,
        }
    (ROOT / "reports/advanced_evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"macro_f1": report["macro_f1"], "mean_task_accuracy": report["mean_task_accuracy"],
                      "pii": tasks["pii"]}, indent=2))


if __name__ == "__main__":
    main()
