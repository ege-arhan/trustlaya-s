# Candidate decisions

Selection used synthetic validation and an internal split of deepset training data. The original synthetic test was not used for candidate selection.

| Candidate | Development result | Decision |
|---|---|---|
| Baseline student | Synthetic validation macro F1 0.487; injection F1 1.000 | Retained |
| 300 additional steps | Same macro F1 0.487 | Rejected: no measured gain |
| Real-data injection head | Internal deepset validation F1 0.840; synthetic injection F1 0.679 | Rejected: false-positive regression |
| Head blend alpha 0.4 | Internal deepset F1 0.780 vs. 0.776 baseline; FPR 0.290 vs. 0.275 | Rejected: marginal gain, worse FPR |
| Language-aware lowercase | Internal deepset F1 0.763 vs. 0.776 baseline | Rejected |

The selection gate favors no material regression on existing tasks. All candidate weights remain local under `models/candidates/` for reproducibility, excluded from publication.
