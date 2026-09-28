"""Generate version-aware, measured edge benchmark tables."""

import csv
import json
from pathlib import Path

import torch

from benchmark_edge import bench

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT / "reports/baseline_report.json").read_text())
    advanced = json.loads((ROOT / "reports/advanced_evaluation.json").read_text())
    parity = json.loads((ROOT / "reports/onnx_v2_parity.json").read_text())
    sizes = json.loads((ROOT / "reports/model_sizes_v2.json").read_text())
    folder = ROOT / "models/trustlaya-s-v2"
    exported = ROOT / "models/exported/v2"
    variants = [
        ("PyTorch CPU", "torch_cpu", None, sizes["student_fp32_bytes"]),
        ("PyTorch MPS", "torch", None, sizes["student_fp32_bytes"]),
        ("ONNX CPU FP32", "onnx", exported / "trustlaya_s.onnx", sizes["onnx_fp32_bytes"]),
        ("ONNX CPU INT8", "onnx_int8", exported / "trustlaya_s_int8.onnx", sizes["onnx_int8_bytes"]),
    ]
    rows = []
    for label, backend, path, size in variants:
        if backend == "torch" and not torch.backends.mps.is_available():
            continue
        measurement = bench(backend, model_dir=folder, onnx_path=path)
        subset = parity["backends"]["int8" if backend == "onnx_int8" else
                                     "fp32" if backend == "onnx" else "torch"]
        tasks = subset["tasks"] if backend == "onnx_int8" else advanced["tasks"]
        rows.append({
            "model": "TrustLaya-S v2 PII candidate", "backend": label,
            "parameters": base["parameters"], "size_mib": round(size / 2**20, 3),
            "macro_f1": round(subset["macro_f1"] if backend == "onnx_int8" else advanced["macro_f1"], 4),
            "security_f1": round(tasks["security_risk"]["f1"], 4),
            "pii_f1": round(tasks["pii"]["f1"], 4),
            "injection_f1": round(tasks["prompt_injection"]["f1"], 4),
            "ece": round(subset["mean_ece"], 4),
            "f1_scope": "synthetic test first 256" if backend == "onnx_int8" else "synthetic full test 1975",
            "ece_scope": "synthetic test first 256",
            "warm_p50_ms": round(measurement["warm_p50_ms"], 3),
            "cold_start_ms": round(measurement["cold_start_ms"], 3),
            "rss_delta_mib_approx": round(measurement["rss_delta_mib_approx"], 1),
        })
    original = base["latency"]
    for label, key, size in (("PyTorch CPU", "pytorch_cpu", base["weight_bytes"]),
                             ("PyTorch MPS", "pytorch_mps", base["weight_bytes"]),
                             ("ONNX CPU FP32", "onnx_cpu", (ROOT / "models/exported/trustlaya.onnx").stat().st_size)):
        if key not in original:
            continue
        measurement = original[key]
        rows.append({
            "model": "TrustLaya-S v1 baseline", "backend": label,
            "parameters": base["parameters"], "size_mib": round(size / 2**20, 3),
            "macro_f1": round(base["macro_f1"], 4),
            "security_f1": round(base["tasks"]["security_risk"]["f1"], 4),
            "pii_f1": round(base["tasks"]["pii"]["f1"], 4),
            "injection_f1": round(base["tasks"]["prompt_injection"]["f1"], 4),
            "ece": None, "f1_scope": "synthetic full test 1975",
            "ece_scope": "NOT MEASURED on same 256 subset",
            "warm_p50_ms": round(measurement["warm_p50_ms"], 3),
            "cold_start_ms": round(measurement["cold_start_ms"], 3),
            "rss_delta_mib_approx": round(measurement["rss_delta_mib_approx"], 1),
        })
    target = ROOT / "benchmarks"
    (target / "results.json").write_text(json.dumps(rows, indent=2))
    with (target / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Edge benchmark comparison", "", "Batch 1 warm p50 from 20 timed calls. V1 latency comes from the fresh frozen baseline run; v2 latency from this run. F1 scopes differ for INT8 and are identified in the CSV/JSON. ECE is available only for a 256-row v2 subset. Sequential RSS deltas are approximate.", "", "| Model | Backend | Size MiB | Macro F1 | Security F1 | PII F1 | Injection F1 | ECE | p50 ms |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {model} | {backend} | {size_mib:.1f} | {macro_f1:.4f} | {security_f1:.4f} | {pii_f1:.4f} | {injection_f1:.4f} | {ece} | {warm_p50_ms:.3f} |".format(
            **{**row, "ece": "NOT MEASURED" if row["ece"] is None else f"{row['ece']:.4f}"}))
    lines += ["", "V2 is a PII-head candidate. The v1 and v2 full synthetic-test F1 scores are unchanged overall; independent Turkish PII results are in `reports/independent_pii_v2.json`. INT8 changed 5.08% of final policy actions on its 256-row subset and remains experimental."]
    (target / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps([{"model": row["model"], "backend": row["backend"],
                       "warm_p50_ms": row["warm_p50_ms"]} for row in rows], indent=2))


if __name__ == "__main__":
    main()
