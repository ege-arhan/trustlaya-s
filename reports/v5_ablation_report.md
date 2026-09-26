# V5 training ablation (8 experiments)

**V5_STATUS = NO_GO.** DEV-selected candidate: `E6_mixed_balanced_focal_510`. Status rules and every threshold,
temperature and checkpoint hash were frozen in [`v5_selection_lock.json`](v5_selection_lock.json)
(SHA-256 `2f719817a0c62380be4c95ebd663a4274cb1f7077ec27203804f0c998ed85e68`) before TEST1/TEST2 were scored once. V2 stays the default; V5 is not
connected to the firewall. No experiment was deleted.

Setup: 42.1M-parameter Turkish BERT (`models/base`, not V2) + one ATTACK logit, full fine-tuning,
AdamW lr 2e-5, weight decay 0.01, batch 16, 3 epochs, 10% warmup + linear decay, seed 42, fp32 on
Apple MPS (not bitwise deterministic). The epoch kept is the lowest shared-DEV BCE loss. Thresholds
are chosen on DEV (macro accuracy over TT-attack, JLL-attack, JLL-benign, docs-benign and arXiv-benign
groups); temperature is fitted on DEV. Context = HEAD content tokens (510 is the maximum; "512" is
not a content length). TEST1 = pool TEST split; TEST2/OOD = deepset + Gandalf. JailbreakBench
(harmful vs benign *requests*) is a side report because it is a different task.

DEV pooled is dominated by Tensor Trust attacks (1,346 of 3,610 rows, attack-only), so read the
source columns. R = recall, FPR = false-positive rate, at each experiment's frozen DEV threshold.

| Experiment | Data | Sampling | Loss | Context | DEV PR-AUC | DEV Recall | DEV FPR | DEV F1 | TEST1 | TEST2/OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| E1_tt_ce_94 | TT + security prose | NATURAL | CE | 94 | 0.894 | 0.898 | 0.283 | 0.786 | TT R 0.975; JLL R 0.384 FPR 0.325; docs FPR 0.010; arXiv FPR 0.000 | deepset R 0.663 FPR 0.662; Gandalf R 0.957 |
| E2_jll_ce_94 | JLL | NATURAL | CE | 94 | 0.606 | 0.071 | 0.027 | 0.128 | TT R 0.036; JLL R 0.448 FPR 0.019; docs FPR 0.010; arXiv FPR 0.141 | deepset R 0.012 FPR 0.008; Gandalf R 0.016 |
| E3_mixed_natural_ce_94 | TT+JLL+security+arXiv | NATURAL | CE | 94 | 0.976 | 0.953 | 0.119 | 0.901 | TT R 0.992; JLL R 0.692 FPR 0.119; docs FPR 0.184; arXiv FPR 0.022 | deepset R 0.857 FPR 0.748; Gandalf R 0.996 |
| E4_mixed_balanced_ce_94 | TT+JLL+security+arXiv | BALANCED_SOURCE | CE | 94 | 0.966 | 0.963 | 0.199 | 0.861 | TT R 0.990; JLL R 0.826 FPR 0.190; docs FPR 0.020; arXiv FPR 0.000 | deepset R 0.806 FPR 0.790; Gandalf R 0.992 |
| E5_mixed_natural_focal_510 | TT+JLL+security+arXiv | NATURAL | FOCAL | 510 | 0.985 | 0.953 | 0.068 | 0.932 | TT R 0.995; JLL R 0.715 FPR 0.073; docs FPR 0.092; arXiv FPR 0.022 | deepset R 0.766 FPR 0.700; Gandalf R 0.997 |
| E6_mixed_balanced_focal_510 | TT+JLL+security+arXiv | BALANCED_SOURCE | FOCAL | 510 | 0.978 | 0.957 | 0.114 | 0.906 | TT R 0.991; JLL R 0.779 FPR 0.115; docs FPR 0.000; arXiv FPR 0.011 | deepset R 0.714 FPR 0.707; Gandalf R 0.994 |
| E7_mixed_balanced_ce_256 | TT+JLL+security+arXiv | BALANCED_SOURCE | CE | 256 | 0.975 | 0.965 | 0.185 | 0.870 | TT R 0.994; JLL R 0.849 FPR 0.169; docs FPR 0.010; arXiv FPR 0.011 | deepset R 0.829 FPR 0.816; Gandalf R 0.993 |
| E8_mixed_balanced_ce_510 | TT+JLL+security+arXiv | BALANCED_SOURCE | CE | 510 | 0.975 | 0.972 | 0.200 | 0.865 | TT R 0.994; JLL R 0.866 FPR 0.183; docs FPR 0.031; arXiv FPR 0.033 | deepset R 0.869 FPR 0.851; Gandalf R 0.996 |
| V2 (frozen, reference) | synthetic | – | – | 94 | – | – | – | – | TT R 0.443; JLL R 0.901 FPR 0.875; docs FPR 0.898; arXiv FPR 0.978 | deepset R 0.865 FPR 0.364; Gandalf R 0.722 |

| Experiment | Checkpoint SHA-256 | Best epoch | Train min | Train rows | DEV threshold | DEV temperature | Generalization failure |
|---|---|---|---|---|---|---|---|
| E1_tt_ce_94 | `b4657e7025db3f72b1938f5b8f98968b8561888f651d33a8dbadb5463e29e010` | 1 | 5.1 | 7947 | 0.99 | 6.2933 | yes |
| E2_jll_ce_94 | `aecfccb0cf8d7e8ade9ca5e87069d2ebb6cd9ec485f91cca7f008e4328a95ff5` | 2 | 5.6 | 9955 | 0.43 | 7.3129 | no |
| E3_mixed_natural_ce_94 | `a7c9e4c33e877d2555782f4c73db5e719e7c3d0aff15981795002a0c938a8ffe` | 1 | 9.9 | 17902 | 0.12 | 1.1533 | yes |
| E4_mixed_balanced_ce_94 | `6b8027de33271cb84e45f82a822f60a81385f1b769f09889877774ea1424965d` | 2 | 9.6 | 17902 | 0.05 | 1.7561 | yes |
| E5_mixed_natural_focal_510 | `114c3573d9591201daa3150814a1e5acdb337db61349db44ba41f945e3eb4ebf` | 3 | 50.4 | 17902 | 0.26 | 0.5868 | yes |
| E6_mixed_balanced_focal_510 | `26468821df242d2f5e873daf2b943dd6787e95cfefa037d8cfacfee2dbe048cb` | 3 | 51.6 | 17902 | 0.34 | 0.6326 | yes |
| E7_mixed_balanced_ce_256 | `8b85d05b4f1786b9dd9fbadb073ce94992a440c9ca4d1749372e3d6790663455` | 3 | 22.7 | 17902 | 0.01 | 1.8648 | yes |
| E8_mixed_balanced_ce_510 | `267983de0445593c9362cc7daa1bba516aaa827c33539553666911b3d907753a` | 3 | 48.7 | 17902 | 0.01 | 1.9217 | yes |

Per-experiment config, data manifest, training log, environment and full metrics (confusion
matrices, PR-AUC, ECE, Brier, per-source and per-length) are in `reports/experiments/<id>/`.
Master file: [`v5_ablation_master.json`](v5_ablation_master.json).

## 1. Tensor Trust contribution

Without Tensor Trust (E2, JLL only) the model catches almost no game attacks: TEST1 Tensor Trust
recall 0.036, Gandalf 0.016, deepset 0.012. With Tensor Trust in the mix, Tensor Trust and Gandalf
recall are 0.99+ in every mixed run (V2: 0.443 and 0.722). Tensor Trust only (E1, negatives from
security prose) transfers badly to JailbreakLLMs (recall 0.384, FPR 0.325).

## 2. JailbreakLLMs contribution

JailbreakLLMs supplies the only large real benign pool (regular prompts). Without it (E1) benign
JailbreakLLMs FPR is 0.325; with it the mixed runs reach 0.073–0.190. Jailbreak recall stays the
hardest target: 0.69–0.87 on TEST1, below V2's 0.901, which V2 reaches only by flagging 87.5% of
regular prompts.

## 3. Source balancing (same data, same budget)

Balanced sampling raised JailbreakLLMs recall and also its FPR: CE/94 0.692→0.826 recall,
0.119→0.190 FPR (E3→E4); focal/510 0.715→0.779 recall, 0.073→0.115 FPR (E5→E6). It was not better on
OOD: deepset FPR 0.748→0.790 (CE/94) and 0.700→0.707 (focal/510). Balanced is not a winner by
default; it trades benign FPR for jailbreak recall. Kish effective sample size and sampling mass
per (source, label) are in each `dataset_manifest.json`.

## 4. CE vs focal (gamma 2.0, alpha 0.25, unchanged)

At 510 tokens with balanced sampling (E8 CE vs E6 focal): focal lowered JailbreakLLMs FPR
0.183→0.115 and deepset FPR 0.851→0.707 but also recall (0.866→0.779 JLL, 0.869→0.714 deepset).
CE thresholds for E7/E8 landed at the grid edge (0.01): CE scores on DEV are compressed near 0 for
many attacks, which is itself a calibration signal.

## 5. 94 vs 256 vs 510 (balanced, CE: E4, E7, E8)

JailbreakLLMs TEST1 recall 0.826 → 0.849 → 0.866; FPR 0.190 → 0.169 → 0.183. Tensor Trust recall is
0.99+ at every length. Longer context helps a little on long jailbreaks and costs 2.4–5x training
time (9.6, 22.7, 48.7 minutes). It does not fix OOD: deepset FPR 0.790 → 0.816 → 0.851.

## 6. Long-context behavior (TEST1 JailbreakLLMs, recall by token-length bucket)

| Experiment | ≤94 | 95–256 | 257–510 | >510 |
|---|---|---|---|---|
| E3 natural CE 94 | 0.833 | 0.807 | 0.686 | 0.638 |
| E4 balanced CE 94 | 0.750 | 0.807 | 0.829 | 0.840 |
| E7 balanced CE 256 | 0.750 | 0.839 | 0.857 | 0.862 |
| E8 balanced CE 510 | 0.750 | 0.871 | 0.886 | 0.872 |
| E6 balanced focal 510 | 0.583 | 0.774 | 0.714 | 0.830 |

E3 loses recall as jailbreaks get longer; reading more tokens (E7/E8) removes most of that drop.
Tensor Trust recall stays 0.98–1.00 in every bucket for all mixed runs. Long benign prompts cost
FPR (JLL >510 bucket FPR 0.25–0.27 for balanced CE).

## 7. Benign false positives

Security documentation FPR 0.000–0.184 and arXiv abstracts 0.000–0.033 for mixed runs (V2: 0.898 and 0.978),
so words like "ignore", "bypass" and "jailbreak" in prose no longer trigger by themselves on these
sources. JailbreakLLMs regular prompts: 0.073–0.190 (V2 0.875). deepset benign: 0.700–0.851 (V2
0.364) — worse than V2.

## 8. Cross-source generalization

Every experiment is flagged GENERALIZATION FAILURE under the pre-registered rule (min attack recall
< 0.5 with max ≥ 0.8, or any benign FPR > 0.20). The single-source runs fail across sources (E1 on
JailbreakLLMs, E2 on Tensor Trust/Gandalf/deepset). All mixed runs fail on deepset benign FPR. Most
deepset rows are short (606 of 648 are ≤94 tokens), so this is a distribution shift, not a context
limit. deepset is partly German and its benign rows look different from forum prompts; which of
these explains the errors was not tested.

## 9. Calibration

DEV-fitted temperatures: mixed CE runs 1.15–1.92, focal runs 0.59–0.63 (focal outputs are
under-confident), E1/E2 6.3–7.3. TEST1 ECE after DEV temperature: 0.015–0.031 for mixed runs
(E6 0.055 raw → 0.015 calibrated). Calibration was never fitted on TEST.

## 10. OOD degradation

Gandalf improves (0.99+ recall vs V2 0.722). deepset degrades: F1 0.505–0.565 vs V2 0.710, driven by
FPR. JailbreakBench (task mismatch) is poor for all runs (E6 recall 0.40, FPR 0.52). A model that
wins on the pool sources and fails on an independent benign set does not meet the status rule
`beats_v2` (every benign-source FPR below V2's), so the status is NO_GO.

## Tokenizer

Training sources are English-dominant: median expansion 2.07 (Tensor Trust) and 2.12
(JailbreakLLMs) tokens per word; 32% and 83% of rows exceed 94 content tokens. Native Turkish text:
1.40 tokens per word, 0.07% over 94. No Turkish row is in any V5 training set, so none of these
results say anything about Turkish.

## What would be next (not run)

1. Add an independent benign source close to deepset's distribution (short instructions/questions,
   including German) to TRAIN, then re-check deepset as OOD with a *different* held-out benign set.
2. Head+tail routing as a separate experiment; the context effect above is modest.
3. Keep these checkpoints as NO_GO; do not publish their weights.
