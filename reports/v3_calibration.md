# V3 calibration and operating points

Calibration was fitted with logistic/Platt scaling on one half of near-duplicate-screened **development** cases/prompts. Thresholds were selected on the other half. TAB was divided by case hash, so its windows do not cross the calibration/selection halves. Gandalf and prompts.chat development prompts were divided by text hash. This is an internal development calibration, not a guarantee of calibrated real-world probability.

| Task | Calibration N | Threshold-selection N | Selected threshold | Selection rule |
|---|---:|---:|---:|---|
| TAB DIRECT PERSON/CODE window presence | 1,038 | 934 | 0.47 calibrated | Highest development F1 with FPR <= 0.10 |
| Gandalf injection vs prompts.chat benign | 222 | 264 | 0.30 calibrated | Highest development F1 with FPR <= 0.25 |

PII token decoding has a separate **0.50 raw entity probability** threshold chosen by development token F1. The token/character-span metric is distinct from the calibrated window-presence score.

## Held-out clean external test

| Task (N) | Score | ECE (10 bins) | Brier | NLL |
|---|---|---:|---:|---:|
| TAB (1,794) | raw | 0.0340 | 0.0550 | 0.1860 |
| TAB (1,794) | calibrated | 0.0264 | 0.0508 | 0.1769 |
| JailbreakLLMs (5,761) | raw | 0.1191 | 0.1414 | 0.4822 |
| JailbreakLLMs (5,761) | calibrated | 0.1212 | 0.1263 | 0.6398 |

TAB calibration improves all three measured metrics slightly, while jailbreak calibration improves Brier but **worsens ECE and NLL** on the independent source. This is another sign that Gandalf calibration does not transfer. A lower Brier score alone is insufficient to call the jailbreak scores calibrated.

The [reliability diagram](v3_reliability.svg) and [bin counts](v3_reliability_bins.json) show the raw and calibrated scores. Complete development threshold curves are in [v3_threshold_curves.json](v3_threshold_curves.json). No final-test threshold was selected. Neither calibrated score is a probability of a legal or ethical verdict.
