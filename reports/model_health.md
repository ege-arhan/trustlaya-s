# TrustLaya-S v2 model health

Research diagnostics only. All performance figures below have their own dataset scope; do not combine them into a production claim.

Synthetic mixed-only test: macro F1 **0.663**, security F1 **0.777**, PII F1 **1.000**, injection F1 **0.555**.

Synthetic validation mean ECE **0.105 raw / 0.089 calibrated**. Mean Brier **0.085**; NLL **0.271**. Temperatures for eight heads were fitted on this same validation set.

Controlled adversarial injection: clean F1 **0.700** (24 rows), transformed F1 **0.750** (216 rows). Worst variant: **encoding**, F1 **0.000**. The transformed aggregate improves because some prefixes make attacks easier; it does not prove robustness.

Separate CC-BY Turkish PII test: v1 F1 **0.737**, v2 F1 **0.784**, v2 FPR **0.316**. Source is synthetic and this set was inspected after training; later tuning against it would bias subsequent claims.

ONNX FP32: **159.9 MiB**, batch-1 CPU p50 **6.185 ms**. INT8 final-action disagreement **0.051** on 256 synthetic rows.

Abstention rate **0.002**. Confidence median **0.999**. Selective action error rises as threshold increases; current confidence is not a valid correctness estimate.

See `reports/model_health.json` for coverage-risk, per-task calibration, size, and all exact measured values.

Independent Apache-2.0 agentic tool-result benchmark (142 attacks, 40 benign): single-window injection F1 **0.484**, recall **0.380**, FPR **0.675**; sliding-window max F1 **0.592**, recall **0.500**, FPR **0.675**. This is text detection only, not agent attack success. Neither setting is adequate for autonomous enforcement. See `reports/agent_injection_independent.json`.
