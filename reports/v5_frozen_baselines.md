# V5 starting point: frozen v2/v3/v4

Exact row predictions and configuration are copied to local read-only `benchmarks/v5/private/frozen/`. The tracked `benchmarks/v5/frozen_manifest.json` stores SHA-256 values. The script rejects changed copies and checks the v2/v3 original freeze. No model or previous report is rewritten.

| Version | JLL clean N | F1 | Recall | FPR | Operating point |
|---|---:|---:|---:|---:|---|
| v2 default | 5,761 | 0.205857 | 0.946457 | 0.897971 | raw 0.50; first 94 content tokens |
| v3 NO-GO | 5,761 | 0.047619 | 0.034646 | 0.052087 | Platt 0.30; first 94 content tokens |
| v4 NO-GO | 5,761 | 0.249945 | 0.900787 | 0.657433 | window max, Platt 0.79 |

V4 has 0.860281 FPR on benign inputs above 510 content tokens. On the separate deepset prompt-injection task it detected 4/60 positives. These are distinct task definitions. JLL was previously inspected, so it is a frozen historical comparison, not a new blind V5 test. The v4 model is local and does not replace v2.
