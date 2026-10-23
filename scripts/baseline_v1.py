"""Measure the frozen v1 checkpoint before Advanced experiments."""

import json
import statistics
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from trustlaya.dataset import read_rows
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS

from benchmark_edge import bench

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = read_rows(ROOT / "data/splits/test.jsonl")
    analyzer = Analyzer("torch_cpu")
    predictions = [analyzer.analyze(row["text"]) for row in rows]
    results = {}
    true_flat, predicted_flat = [], []
    for task in TASKS:
        true = [row["labels"][task] for row in rows]
        predicted = [int(item[task] >= 0.5) for item in predictions]
        precision, recall, f1, _ = precision_recall_fscore_support(
            true, predicted, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(true, predicted, labels=[0, 1]).ravel()
        results[task] = {
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "accuracy": float(accuracy_score(true, predicted)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "support_positive": int(sum(true)),
        }
        true_flat.extend(true)
        predicted_flat.extend(predicted)
    micro = precision_recall_fscore_support(
        true_flat, predicted_flat, average="binary", zero_division=0
    )
    import torch

    latency = {"pytorch_cpu": bench("torch_cpu"), "onnx_cpu": bench("onnx")}
    if torch.backends.mps.is_available():
        latency["pytorch_mps"] = bench("torch")
    weights = ROOT / "models/student/model.safetensors"
    tokenizer_files = [ROOT / "models/student" / name for name in
                       ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")]
    report = {
        "checkpoint": str(weights), "test_rows": len(rows),
        "parameters": sum(p.numel() for p in analyzer.model.parameters()),
        "trainable_parameters_recorded": json.loads((ROOT / "reports/training.json").read_text())["trainable_parameters"],
        "weight_bytes": weights.stat().st_size,
        "tokenizer_bytes": sum(p.stat().st_size for p in tokenizer_files),
        "max_sequence_length": 96,
        "mean_task_accuracy": statistics.mean(x["accuracy"] for x in results.values()),
        "macro_f1": statistics.mean(x["f1"] for x in results.values()),
        "micro_f1": float(micro[2]), "tasks": results, "latency": latency,
    }
    dest = ROOT / "reports/baseline_report.json"
    dest.write_text(json.dumps(report, indent=2))
    lines = ["# Frozen v1 baseline", "", "Measured on the existing 1,975-row **synthetic** test split. This split is mixed-language only; dataset audit explains the resulting confound. Scores include deterministic PII/secret evidence floors. Micro F1 pools positive decisions across all nine binary tasks.", "", f"Parameters: {report['parameters']:,}; recorded trainable parameters: {report['trainable_parameters_recorded']:,}. Weights: {report['weight_bytes'] / 2**20:.2f} MiB; tokenizer files: {report['tokenizer_bytes'] / 2**20:.2f} MiB; sequence length: 96.", "", f"Mean task accuracy: {report['mean_task_accuracy']:.4f}; macro F1: {report['macro_f1']:.4f}; micro F1: {report['micro_f1']:.4f}.", "", "| Task | Precision | Recall | F1 | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, row in results.items():
        (tn, fp), (fn, tp) = row["confusion_matrix"]
        lines.append(f"| {task} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {tn} | {fp} | {fn} | {tp} |")
    lines += ["", "Batch-one warm p50 latency (20 timed calls after 3 warmups):", "", "| Backend | p50 ms | Cold start ms | Approx RSS delta MiB |", "|---|---:|---:|---:|"]
    for backend, row in latency.items():
        lines.append(f"| {backend} | {row['warm_p50_ms']:.3f} | {row['cold_start_ms']:.3f} | {row['rss_delta_mib_approx']:.1f} |")
    lines += ["", "RSS deltas are sequential and approximate. All metrics are measured on the local MacBook; they do not establish production performance. Machine-readable values: `reports/baseline_report.json`."]
    (ROOT / "reports/baseline_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: report[k] for k in ("test_rows", "mean_task_accuracy", "macro_f1", "micro_f1")}, indent=2))


if __name__ == "__main__":
    main()
