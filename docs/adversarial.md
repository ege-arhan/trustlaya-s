# Adversarial evaluation

`scripts/evaluate_adversarial.py` evaluates 24 researcher-authored base cases, each with nine transformations: paraphrase, typo, Unicode, spacing, encoding, translation, mixed language, indirect wording, and role manipulation. The 240-row benchmark is controlled synthetic data, not a human-verified attack corpus. Related variants share a family and must never be split across train and test.

Measured prompt-injection F1 is 0.700 on 24 originals and 0.750 on 216 transformed cases. This aggregate is misleading: encoding has F1 **0.000** and false-negative rate **1.000** on 12 positive cases. Indirect wording has false-positive rate 0.667 on 12 negatives; mixed language has 0.417. Full counts and rates are in `reports/adversarial.json`.

Reproduce with `python scripts/evaluate_adversarial.py`. New external corpora must be checked for license, duplicates, task definition, and contamination before use. Decoded or tool-returned text should be inspected by a separate trust-boundary policy; the current text classifier does not reliably recognize encoded attacks.
