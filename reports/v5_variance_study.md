# V5 variance study: seeds, batching and the threshold rule

Descriptive only: no model, threshold or setting was selected from these numbers. All TEST sets
were scored in earlier iterations, so this measures run-to-run variation, not a new blind result.
Every run uses the iteration-2 DEV threshold rule on the iteration-2 DEV. Numbers:
[`v5_variance_study.json`](v5_variance_study.json).

| Config (seeds) | TT recall | JLL recall | JLL FPR | deepset recall | deepset FPR | deepset PR-AUC | MASSIVE FPR (en / de / tr) | TEST1 ECE (cal.) |
|---|---|---|---|---|---|---|---|---|
| E6 data, random batches (42, 43, 44) | 0.931 ± 0.030 | 0.395 ± 0.057 | 0.014 | 0.246 ± 0.048 | 0.120 ± 0.050 | 0.510 | 0.255 ± 0.117 (0.29 / 0.27 / 0.21) | 0.114 |
| E6 data, grouped batches (42, 43, 44) | 0.911 ± 0.007 | 0.411 ± 0.069 | 0.017 | 0.218 ± 0.028 | 0.139 ± 0.036 | 0.484 | 0.305 ± 0.112 (0.34 / 0.30 / 0.27) | 0.097 |
| A4 data, grouped batches (42, 43, 44) | 0.984 ± 0.001 | 0.781 ± 0.050 | 0.144 | 0.235 ± 0.030 | 0.067 ± 0.009 | 0.626 | 0.185 ± 0.012 (0.29 / 0.13 / 0.14) | 0.022 |
| A4 data, random batches (42) | 0.983 | 0.750 | 0.143 | 0.242 | 0.040 | 0.653 | 0.191 (0.28 / 0.14 / 0.16) | 0.017 |

Seed 42 of "E6 data, random" is the original E6 checkpoint; seed 42 of the grouped rows are R0 and A4.

## Findings

1. **Batching is not the cause.** Random and length-grouped batches give overlapping results on
   the same data (JLL recall 0.395 vs 0.411, deepset recall 0.246 vs 0.218; the A4 random run sits
   inside the A4 grouped range).
2. **The iteration-2 "collapse" was mainly the threshold rule.** The original E6 checkpoint scores
   deepset recall 0.714 at its iteration-1 threshold (0.34), but 0.298 at the threshold the
   iteration-2 rule picks for it on the expanded DEV (0.73). The iteration-2 rule averages over six
   benign and two attack groups, which pushes thresholds up.
3. **Adding multilingual benign data helps and is stable.** At the same rule, A4 data vs E6 data
   (3 seeds each): JLL recall 0.40 → 0.78, deepset FPR 0.12 → 0.07, MASSIVE FPR 0.26 → 0.19 with
   much lower seed spread, deepset PR-AUC 0.51 → 0.63, calibrated ECE 0.11 → 0.02.
4. **deepset recall is a ranking problem.** A post-hoc class-balanced threshold raises deepset
   recall to 0.30–0.38 for A4 seeds but also raises MASSIVE FPR to 0.32–0.44. No threshold separates
   deepset attacks from short benign requests well; the model needs better attack examples of that
   kind (iteration 3, HackAPrompt).
5. **Seed spread is large for E6 data** (MASSIVE FPR sd 0.12, Turkish 0.04–0.43), so single-seed
   comparisons in earlier reports are indicative only.
