# Edge benchmark comparison

Batch 1 warm p50 from 20 timed calls. V1 latency comes from the fresh frozen baseline run; v2 latency from this run. F1 scopes differ for INT8 and are identified in the CSV/JSON. ECE is available only for a 256-row v2 subset. Sequential RSS deltas are approximate.

| Model | Backend | Size MiB | Macro F1 | Security F1 | PII F1 | Injection F1 | ECE | p50 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TrustLaya-S v2 PII candidate | PyTorch CPU | 160.8 | 0.6632 | 0.7769 | 1.0000 | 0.5549 | 0.1128 | 12.087 |
| TrustLaya-S v2 PII candidate | PyTorch MPS | 160.8 | 0.6632 | 0.7769 | 1.0000 | 0.5549 | 0.1128 | 4.931 |
| TrustLaya-S v2 PII candidate | ONNX CPU FP32 | 159.9 | 0.6632 | 0.7769 | 1.0000 | 0.5549 | 0.1128 | 4.552 |
| TrustLaya-S v2 PII candidate | ONNX CPU INT8 | 40.5 | 0.6320 | 0.7943 | 1.0000 | 0.5758 | 0.1076 | 5.190 |
| TrustLaya-S v1 baseline | PyTorch CPU | 160.8 | 0.6632 | 0.7769 | 1.0000 | 0.5549 | NOT MEASURED | 12.656 |
| TrustLaya-S v1 baseline | PyTorch MPS | 160.8 | 0.6632 | 0.7769 | 1.0000 | 0.5549 | NOT MEASURED | 5.541 |
| TrustLaya-S v1 baseline | ONNX CPU FP32 | 159.9 | 0.6632 | 0.7769 | 1.0000 | 0.5549 | NOT MEASURED | 5.140 |

V2 is a PII-head candidate. The v1 and v2 full synthetic-test F1 scores are unchanged overall; independent Turkish PII results are in `reports/independent_pii_v2.json`. INT8 changed 5.08% of final policy actions on its 256-row subset and remains experimental.
