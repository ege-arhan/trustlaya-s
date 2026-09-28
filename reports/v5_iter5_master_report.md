# V5 iteration 5 master report: controlled synthetic multilingual augmentation

**V5_STATUS = NO_GO.** DEV-selected candidate S2_de (German machine-translated attacks). Rules
(fixed in [iter5_plan.md](iter5_plan.md)): A deepset recall ≥ 0.40 ✗ (0.398), B deepset FPR ≤ 0.15 ✓,
C MASSIVE not worse than S0 ✗ (dev 0.247 → 0.369, test 0.262 → 0.375), D Tensor Trust ✓, E
JailbreakLLMs ✓, G seed spread ✓, F not assessable (no real multilingual attack source).
**V2 stays the default and the firewall model; no V5 checkpoint is published or connected.**
Synthetic rows were TRAIN-only; DEV and every TEST set contain none (asserted in code). Thresholds,
temperatures and checkpoints were locked in [`v5_iter5_lock.json`](v5_iter5_lock.json) before one
TEST run. deepset, Gandalf and MASSIVE were scored in earlier iterations (reused, not blind).

## Answer to the main question

**No. Turkish/German synthetic attack augmentation did not generalize; it taught a language
shortcut.** Adding German translated attacks raised deepset German attack recall (S0 0.10–0.14 →
S2 0.41–0.49) but raised false alarms on real German benign requests in MASSIVE dev from 0.16–0.19
to 0.63–0.72. Turkish synthetic attacks did the same for Turkish (MASSIVE-dev tr 0.19–0.25 → 0.60–0.69).
deepset PR-AUC did not move (S0 0.670, S2 0.671, S1 0.675): the model did not separate attacks from
benign text better, it moved all German (or Turkish) text toward "attack". Translated benign
controls (S4, S5) reduced but did not remove the shift (S4 MASSIVE-dev de 0.52, tr 0.36).

## Config means (3 seeds; S0 = iteration-3 H1 recipe)

| Config | deepset recall | deepset FPR | deepset PR-AUC | MASSIVE-dev FPR | MASSIVE-test FPR | TT recall | JLL recall | Gandalf recall | TEST1 ECE (cal.) |
|---|---|---|---|---|---|---|---|---|---|
| S0 | 0.299 ± 0.039 | 0.070 ± 0.013 | 0.670 ± 0.018 | 0.247 ± 0.044 | 0.262 ± 0.043 | 0.988 ± 0.003 | 0.828 ± 0.026 | 0.978 ± 0.008 | 0.017 ± 0.003 |
| S2_de | 0.398 ± 0.030 | 0.115 ± 0.015 | 0.671 ± 0.011 | 0.369 ± 0.060 | 0.375 ± 0.059 | 0.987 ± 0.004 | 0.824 ± 0.020 | 0.972 ± 0.008 | 0.017 ± 0.002 |
| S1_tr | 0.327 ± 0.022 | 0.083 ± 0.024 | 0.675 ± 0.025 | 0.425 ± 0.048 | 0.446 ± 0.058 | 0.990 ± 0.002 | 0.862 ± 0.017 | 0.978 ± 0.004 | 0.013 ± 0.001 |

## Every run (each at its locked DEV threshold)

deepset German/English uses a stopword language guess (*). Question = ends with "?" or starts
with a question word.

| Run | Threshold | deepset recall | deepset FPR | deepset PR-AUC | deepset recall de* / en* | deepset recall question / non-question | MASSIVE-dev FPR en / de / tr | TT recall | JLL recall | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|
| S0_s42 | 0.37 | 0.254 | 0.056 | 0.651 | 0.096 / 0.289 | 0.085 / 0.434 | 0.262 / 0.157 / 0.189 | 0.984 | 0.802 | `617c678ad82b…` |
| S0_s43 | 0.24 | 0.325 | 0.081 | 0.672 | 0.120 / 0.382 | 0.131 / 0.533 | 0.427 / 0.194 / 0.253 | 0.990 | 0.826 | `d89341018065…` |
| S0_s44 | 0.3 | 0.318 | 0.073 | 0.688 | 0.145 / 0.355 | 0.169 / 0.475 | 0.329 / 0.175 / 0.242 | 0.990 | 0.855 | `7b054caaf3be…` |
| S1_tr_s42 | 0.21 | 0.333 | 0.109 | 0.661 | 0.133 / 0.395 | 0.131 / 0.549 | 0.446 / 0.266 / 0.691 | 0.992 | 0.872 | `f02b5c19ea99…` |
| S1_tr_s43 | 0.25 | 0.302 | 0.078 | 0.661 | 0.120 / 0.355 | 0.123 / 0.492 | 0.321 / 0.183 / 0.616 | 0.989 | 0.872 | `7fa39aec5842…` |
| S1_tr_s44 | 0.28 | 0.345 | 0.061 | 0.704 | 0.205 / 0.368 | 0.162 / 0.541 | 0.397 / 0.312 / 0.604 | 0.989 | 0.843 | `c10cf390dbab…` |
| S2_de_s42 | 0.41 | 0.369 | 0.099 | 0.661 | 0.410 / 0.303 | 0.146 / 0.607 | 0.188 / 0.626 / 0.097 | 0.983 | 0.802 | `3d6f2810778a…` |
| S2_de_s43 | 0.27 | 0.429 | 0.129 | 0.669 | 0.494 / 0.349 | 0.177 / 0.697 | 0.322 / 0.725 / 0.219 | 0.990 | 0.843 | `d96df474f120…` |
| S2_de_s44 | 0.34 | 0.397 | 0.116 | 0.683 | 0.470 / 0.309 | 0.177 / 0.631 | 0.307 / 0.700 / 0.126 | 0.988 | 0.826 | `3456db1f92ab…` |
| S3_tr_de_s42 | 0.23 | 0.436 | 0.144 | 0.663 | 0.530 / 0.336 | 0.185 / 0.705 | 0.363 / 0.807 / 0.638 | 0.991 | 0.861 | `667bfa30ed9a…` |
| S4_tr_de_benign_s42 | 0.29 | 0.397 | 0.083 | 0.683 | 0.458 / 0.316 | 0.146 / 0.664 | 0.314 / 0.516 / 0.360 | 0.989 | 0.820 | `65870ab068a0…` |
| S5_de_debenign_s42 | 0.29 | 0.409 | 0.114 | 0.688 | 0.482 / 0.336 | 0.200 / 0.631 | 0.382 / 0.528 / 0.183 | 0.992 | 0.814 | `c3c2ac8a9170…` |

Effect isolation (seed 42): Turkish synthetic (S1) barely changes deepset (0.254 → 0.333) and
raises Turkish benign FPR; German synthetic (S2) raises deepset German recall and German benign
FPR; both (S3) give the highest deepset recall (0.436) and the worst MASSIVE-dev FPR (0.603;
de 0.81). Question-form deepset attacks stay at 0.08–0.20 in every run: the translated attacks are
rarely questions (audit: ≈5%), so this augmentation could not address that failure.

## Data, audit and leakage

- Synthetic data: [v5_iter5_synthetic_data.md](v5_iter5_synthetic_data.md) (S3 synthetic share of
  attack TRAIN 12.8%, below the 15% limit); provenance per row in `data/v5_iter5_manifest.json`.
- Translation audit: [v5_iter5_translation_audit.md](v5_iter5_translation_audit.md) (back-translation
  chrF median de 0.74, tr 0.67; question form kept in about 68%; translated benign controls are
  mostly questions, translated attacks rarely are).
- Leakage: 2 synthetic rows overlapped MASSIVE and were dropped; English originals come only from
  TRAIN. Seed/length details: [`v5_iter5_seed_variance.json`](v5_iter5_seed_variance.json),
  [`v5_iter5_language_generalization.json`](v5_iter5_language_generalization.json),
  [`v5_iter5_short_attack_analysis.json`](v5_iter5_short_attack_analysis.json).
- DEV selected S2_de and S1_tr by class-balanced accuracy differences of ≤ 0.004; DEV has no German
  or Turkish attacks, so it could not see the shortcut.

## Conclusions

1. Machine-translated or author-curated synthetic attacks in a language, without real benign text
   of the same *style* in that language, make language itself an attack cue.
2. The deepset gap needs real, human-written German attacks and question-shaped attacks; translation
   of game-style English attacks does not supply either.
3. V2 remains the default. Iterations 2–5 are recorded as research; no V5 checkpoint is a release
   candidate.
