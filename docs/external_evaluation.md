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

## Independent Turkish PII and multilingual secret diagnostics

A deterministic seed-42, balanced 2,000-example evaluation was run on each source. No rows were used for training or temperature fitting, and no raw rows or detected secret spans are redistributed. Results: `reports/external_privacy_secret.json` and reproducible script `scripts/evaluate_privacy_secret_external.py`.

| Source and scope | Model F1 / recall / FPR | Rule F1 / recall / FPR | Hybrid F1 / recall / FPR |
|---|---|---|---|
| [Turkish privacy filter dataset](https://huggingface.co/datasets/yusuf-said/turkish-privacy-filter-dataset), supported PII categories only; 1,000 positive and 1,000 negative | 0.803 / 0.735 / 0.096 | 0.604 / 0.433 / 0.000 | **0.880 / 0.862 / 0.096** |
| [Prowl secrets corpus](https://huggingface.co/datasets/Podric/prowl-secrets-corpus), synthetic/augmented rows only; 1,000 positive and 1,000 negative | 0.676 / 0.994 / **0.949** | 0.546 / 0.403 / 0.073 | 0.676 / 0.995 / **0.950** |

The Turkish privacy dataset is itself synthetic/curated and includes dates and URLs outside our PII task. These are excluded from the binary ground truth here. The hybrid PII result demonstrates that the regex layer rescues some external misses, while names and account numbers remain weak for exact span extraction. On the independent secret corpus, the model fires on hashes, Git SHAs, IDs and other benign high-entropy strings. A high model-only secret score therefore routes to REVIEW; only verified secret evidence can trigger an immediate secret BLOCK. This policy change limits the impact of model shift but still leaves considerable review workload. The secret corpus is CC BY-NC 4.0 and is used for diagnostics only.

A separate, seed-42 development/test threshold diagnostic on the same Prowl source (`reports/secret_threshold_diagnostic.json`) yielded secret-score test ROC AUC **0.536**. A development-selected threshold near 1.0 still produced test FPR **0.837** with recall **0.941**. Its test subset is separate from the 2,000-row table above. This rules out a simple threshold adjustment as an adequate repair.
