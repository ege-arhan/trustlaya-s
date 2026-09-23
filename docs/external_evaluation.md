# Independent data evaluation

These sources were never used in the original student training or temperature fitting. This table is task-specific: labels cannot be merged into one safety score. Aggregate results are in `reports/external_evaluation.json`.

| Source | Test surface | Result | Interpretation |
|---|---|---|---|
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | 116 held-out injection/benign texts | F1 0.765; recall 0.867; false-positive rate 0.429 at score 0.5 | Independent prompt-injection benchmark. Class balance and text style differ from production. |
| [Rogue Security real-world benign use cases](https://huggingface.co/datasets/rogue-security/real-world-benign-use-cases) | 178 hard, production-derived benign agent texts | 43 false positives; rate 0.242 at score 0.5 | Intentionally selected hard negatives. This is not an overall production false-positive estimate. License is marked `other`; raw data is not redistributed. |
| [Overfit-GM Turkish toxic language](https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language) | Random 2,000-row sample, toxic vs. not toxic | Ethics-risk proxy F1 0.012; recall 0.006 | Toxicity and ethics risk are different tasks. This result defines a scope gap, not an ethics benchmark. Source labels include pseudo labels. |
| [GenAI incidents](https://huggingface.co/datasets/emmanuelgjr/genai-incidents) | 300 reviewed, real-world incident descriptions | Any-risk detection 1.000; REVIEW/BLOCK/REDACT 1.000 | Positive-only diagnostic. With no matched benign incident-style controls, this does not prove useful discrimination. Only aggregate results are saved; source has row-level attribution duties. |

## Experiments

- Extra 300 steps from baseline: synthetic validation macro F1 remained 0.487. Rejected.
- Fine-tune injection head on deepset train only: best internal validation F1 0.840; original synthetic validation injection F1 fell from 1.000 to 0.679 due to false positives. Rejected.
- Blend baseline and fine-tuned injection heads: best eligible blend had internal F1 0.780 versus baseline 0.776 and higher false-positive rate (0.290 vs. 0.275); gain too small to justify change. Rejected.
- Language-aware English lowercasing: external internal validation F1 declined from 0.776 to 0.763. Rejected.
- Threshold sweep on deepset train: F1-optimal threshold 0.65; held-out test F1 0.770 versus 0.765 at 0.5. False-positive rate remained 0.411. Threshold adjustment alone does not solve the problem; deployment policy remains unchanged.

Original synthetic test was not used to select these candidates. The deepset test and Rogue benign sets are reported as external diagnostics. Improving prompt-injection false positives requires human-labeled hard negatives and a separate final holdout.
