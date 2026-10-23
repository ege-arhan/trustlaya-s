# External calibration diagnostic

Raw probabilities and the existing internally fitted calibration were evaluated on the external test sets. **No temperature was fitted on either external test set.** These probabilities are task-model scores, not verified real-world risk likelihoods.

| Dataset | Score | ECE (10 bins) | Brier | NLL |
|---|---|---:|---:|---:|
| TAB | Raw | 0.148 | 0.147 | 0.688 |
| TAB | Existing internal temperature | 0.117 | 0.138 | 0.495 |
| JailbreakLLMs | Raw | 0.736 | 0.677 | 2.274 |
| JailbreakLLMs | Existing internal temperature | 0.799 | 0.790 | 7.224 |

The existing temperature does not change 0.50 classifications here. On this domain-shifted jailbreak set it worsens all three calibration metrics. The 10-bin reliability data below allow a diagram to be reproduced without plotting raw text.

| Dataset | Bin | N | Mean raw score | Observed positive rate |
|---|---:|---:|---:|---:|
| TAB | 0.0–0.1 | 1704 | 0.014 | 0.106 |
| TAB | 0.1–0.2 | 96 | 0.142 | 0.083 |
| TAB | 0.2–0.3 | 53 | 0.249 | 0.057 |
| TAB | 0.3–0.4 | 33 | 0.341 | 0.212 |
| TAB | 0.4–0.5 | 18 | 0.444 | 0.000 |
| TAB | 0.5–0.6 | 25 | 0.548 | 0.120 |
| TAB | 0.6–0.7 | 20 | 0.661 | 0.100 |
| TAB | 0.7–0.8 | 28 | 0.741 | 0.143 |
| TAB | 0.8–0.9 | 23 | 0.857 | 0.130 |
| TAB | 0.9–1.0 | 79 | 0.972 | 0.114 |
| JailbreakLLMs | 0.0–0.1 | 221 | 0.033 | 0.045 |
| JailbreakLLMs | 0.1–0.2 | 75 | 0.147 | 0.120 |
| JailbreakLLMs | 0.2–0.3 | 85 | 0.251 | 0.071 |
| JailbreakLLMs | 0.3–0.4 | 93 | 0.351 | 0.043 |
| JailbreakLLMs | 0.4–0.5 | 87 | 0.449 | 0.069 |
| JailbreakLLMs | 0.5–0.6 | 127 | 0.555 | 0.055 |
| JailbreakLLMs | 0.6–0.7 | 161 | 0.653 | 0.056 |
| JailbreakLLMs | 0.7–0.8 | 311 | 0.755 | 0.039 |
| JailbreakLLMs | 0.8–0.9 | 852 | 0.861 | 0.029 |
| JailbreakLLMs | 0.9–1.0 | 3876 | 0.957 | 0.141 |

## Post-baseline TAB dev experiment

A temperature of 2.252 was fitted on 1896 official TAB dev windows after training-data and test-window exact/near overlap filtering. 12 exact and 242 near dev/test matches were removed without consulting test labels. This did not modify deployed model files.

| Split | Score | ECE | Brier | NLL |
|---|---|---:|---:|---:|
| Dev | Raw | 0.094 | 0.102 | 0.486 |
| Dev | Dev-fitted | 0.110 | 0.098 | 0.352 |
| Test | Raw | 0.148 | 0.147 | 0.688 |
| Test | Dev-fitted | 0.091 | 0.133 | 0.444 |

JailbreakLLMs has no official held-out calibration split, so no new external temperature was fitted there. TAB test labels were never used to fit the dev-only calibrator.
