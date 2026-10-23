# V3 duplicate and leakage audit

Source commits and local hashes are in [training data](v3_training_data.md). We screened case/window or prompt text using exact bytes after whitespace/case normalization and 4–5 character n-gram cosine similarity >= 0.85. The threshold was fixed before examining v3 test metrics. TAB official cases are partitioned by case ID, but repeated legal boilerplate and case references create similar windows across partitions. Exclusion is deliberately conservative; similarity is not proof that a gold entity was memorized.

| Comparison | Reference / target | Exact target matches | Exact or near target matches |
|---|---:|---:|---:|
| TAB train → dev | 6,121 / 2,157 windows | 9 | 185 |
| TAB train → frozen test | 6,121 / 2,079 | 3 | 196 |
| TAB dev → frozen test | 2,157 / 2,079 | 12 | 242 |
| Gandalf + prompts.chat train → development | 2,056 / 490 prompts | 0 | 4 |
| Prompt train → frozen JailbreakLLMs test | 2,056 / 5,888 | 1 | 110 |
| Prompt development → frozen JailbreakLLMs test | 490 / 5,888 | 1 | 25 |
| Prompt train → Gandalf official test | 2,056 / 112 | 0 | 0 |
| Prompt development → Gandalf official test | 490 / 112 | 0 | 0 |

The union excluded **285 TAB test windows**, leaving **1,794** (103 positive), and **127 jailbreak test prompts**, leaving **5,761** (635 positive). All reported v3 headline metrics and paired v2 comparisons use exactly these same retained rows. TAB train-overlapping development windows (185) and four prompt train-overlapping development prompts were excluded before v3 calibration or threshold selection. The frozen v2 2,079/5,888-row baseline and its predictions remain intact.

Normalized exact overlap of v3 training text with the pinned v2 project/Turkish training text was **0/6,121 TAB windows** and **0/2,056 prompt rows**. The previous frozen v2 evaluation had already screened its own test rows against v2 training text. A new char-ngram near-duplicate screen of v3 train against v2 train was not completed; encoder pretraining and semantic overlap cannot be ruled out. Exact source row membership and near-duplicate IDs are in `reports/v3_leakage_results.json`; it contains IDs and similarity values, no raw text.

The first v3 candidate was evaluated once on the frozen external test and failed the attack-transfer criterion. Subsequent TAB code-span postprocessing was selected after inspecting **development** examples, but the clean TAB test outcome was already known. This means the final v3 PII estimate is an iterative experimental estimate, not a pristine one-shot blind holdout. A new independent holdout is required before a release claim.
