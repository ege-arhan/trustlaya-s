# V5 evidence gates after partial Council review

| Gate | Current state | Required artifact and pass condition |
|---|---|---|
| 0. Candidate freeze | **Done locally** | [600-row packet manifest](../benchmarks/v5/review_packet_manifest.json) and [v2–v4 freeze](../benchmarks/v5/frozen_manifest.json) verify unchanged private text/templates and baseline weights/predictions. Tensor Trust data rights remain unresolved. |
| 1. Human GOLD | **Pending, 0/600** | Two distinct human reviews on every GOLD row, independent senior resolution, Level 1 nominal alpha reported (target ≥0.80), unresolved rows excluded, and adequate per-source/per-intent counts. A packet checksum alone is not GOLD. |
| 2. Context/tokenizer diagnosis | **Exploratory done, GOLD repeat pending** | [94/510-token ablation](v5_native_context_diagnostic.md) and [tokenizer audit](v5_tokenizer_audit.md) exist on unlabeled/source-derived data. Repeat with reviewed attack locations and benign discussion slices. |
| 3. Calibrated DEV baseline | **Pending** | Source-separated human DEV; strategy, threshold and calibration fixed there; PR-AUC, FNR at low FPR, FPR at high recall, ECE/Brier/NLL and per-source uncertainty reported. No final test optimization. |
| 4. V5 model training | **Not authorized by evidence yet** | Train only if gate 3 shows read strategy alone is insufficient and label/data rights are resolved. Preserve v2/v3/v4 checkpoints. |
| 5. Hidden transfer | **Pending** | Two untouched sources, exact paired v2/v4/V5 predictions, contamination removal, attack/discussion/long-context slices, fixed operating point, and documented false positives/negatives. |
| 6. Gateway decision | **Pending** | Regression tests, fail-closed authorization, model/policy/gateway attribution, and operational false allow/false block based on reviewed examples. |
| 7. UNO Q | **Not tested** | Physical board ONNX/INT8 correctness, latency, peak RAM, quantization loss, and end-to-end enforcement measurement. No Mac timing is an UNO Q result. |

Council thresholds such as ECE ≤0.08, FAR ≤0.001, board latency ≤150 ms and <15% out-of-domain loss are **proposed targets**, not achieved results or universal acceptance criteria. Set final operational limits from an explicit threat model and measured board capacity. Stop at gate 3 before training, as instructed in the V5 milestone.
