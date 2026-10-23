# Frozen v1 baseline

Measured on the existing 1,975-row **synthetic** test split. This split is mixed-language only; dataset audit explains the resulting confound. Scores include deterministic PII/secret evidence floors. Micro F1 pools positive decisions across all nine binary tasks.

Parameters: 42,138,641; recorded trainable parameters: 19,185,681. Weights: 160.76 MiB; tokenizer files: 0.98 MiB; sequence length: 96.

Mean task accuracy: 0.9066; macro F1: 0.6632; micro F1: 0.7497.

| Task | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| pii | 1.0000 | 1.0000 | 1.0000 | 1619 | 0 | 0 | 356 |
| secret | 1.0000 | 1.0000 | 1.0000 | 1774 | 0 | 0 | 201 |
| prompt_injection | 0.4828 | 0.6523 | 0.5549 | 1501 | 195 | 97 | 182 |
| dangerous_instruction | 1.0000 | 0.4949 | 0.6621 | 1779 | 0 | 99 | 97 |
| privacy_risk | 0.6336 | 0.4579 | 0.5316 | 1569 | 85 | 174 | 147 |
| security_risk | 0.6753 | 0.9145 | 0.7769 | 341 | 499 | 97 | 1038 |
| ethics_risk | 1.0000 | 0.2845 | 0.4429 | 1634 | 0 | 244 | 97 |
| oversight_risk | 1.0000 | 1.0000 | 1.0000 | 1606 | 0 | 0 | 369 |
| data_governance_risk | 0.0000 | 0.0000 | 0.0000 | 1804 | 0 | 171 | 0 |

Batch-one warm p50 latency (20 timed calls after 3 warmups):

| Backend | p50 ms | Cold start ms | Approx RSS delta MiB |
|---|---:|---:|---:|
| pytorch_cpu | 12.656 | 471.891 | 148.7 |
| onnx_cpu | 5.140 | 117.981 | 356.4 |
| pytorch_mps | 5.541 | 647.419 | 31.3 |

RSS deltas are sequential and approximate. All metrics are measured on the local MacBook; they do not establish production performance. Machine-readable values: `reports/baseline_report.json`.
