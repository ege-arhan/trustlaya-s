# V5 iteration 4 master report

**V5_STATUS = NO_GO for iteration 4.** No new model was trained: the public-data search found no
real, human-written Turkish or German attack source and no independent question-form attack
source, and the rules forbid starting synthetic training in this iteration. What was done instead:
the search itself, and a DEV-selected threshold ablation on the iteration-3 checkpoints. **V2 stays
the default and the firewall model. No V5 checkpoint is published or connected to anything.**
Earlier results (V2, E6, R0, A4, H1) are unchanged.

## Evidence gates

| Gate | Result | Evidence |
|---|---|---|
| 1 Real Turkish/German attack source | **FAIL** | [dataset research](v5_iter4_dataset_research.md): all Turkish sets are synthetic, machine-translated or derived from deepset; German exists only as machine translation |
| 2 Short question-form attack source | **PARTIAL** | HackAPrompt is short and human but rarely question-shaped; no independent question-form set |
| 3 No train/test leakage | PASS | no new data; existing checksum/cluster/leakage tests pass; `beratcmn/turkish-prompt-injections` (deepset translation) and zn-bench (contains deepset) rejected |
| 4 Benign FPR falls | PASS (earlier iterations) | fresh MASSIVE-dev FPR: A4 0.441 → H1 0.247 (iteration 3, T1) |
| 5 deepset attack recall rises | **FAIL** | best DEV-selected threshold raises it to 0.384 only by doubling benign FPR (below) |
| 6 Other independent OOD attack recall kept | PASS | Gandalf 0.96–0.99 for every strategy |
| 7 Tensor Trust recall kept | PASS | 0.98–0.99 |
| 8 JailbreakLLMs recall kept | PASS | 0.73–0.91 depending on threshold |
| 9 Calibration kept | PASS | iteration-3 TEST1 ECE 0.017 after DEV temperature |
| 10 Seed variance acceptable | PASS | H1 deepset recall sd 0.04, fresh MASSIVE FPR sd 0.04 (3 seeds) |

## Threshold ablation (DEV selects; TEST reused from earlier iterations)

Strategies: T1 class-balanced global threshold (iteration-3 rule); T2 T1 with DEV short-attack
(≤94 tokens) recall ≥ 0.95; T3 attack groups vs language-balanced benign accuracy; T4 threshold 0.5
on DEV-temperature-calibrated probability (temperature scaling leaves 0.5 in place, so T4 equals a
raw 0.5 threshold). The selection criterion, fixed before scoring, is the worst DEV stratum
(attack recall per length bucket and for question-form attacks, benign accuracy per language).
Mean ± sd over seeds 42/43/44. Full numbers: [`v5_iter4_threshold_ablation.json`](v5_iter4_threshold_ablation.json).

| Model / strategy | DEV worst stratum | deepset recall | deepset FPR | deepset recall de* | deepset recall questions | fresh MASSIVE-dev FPR | Gandalf recall | TT recall | JLL recall / FPR |
|---|---|---|---|---|---|---|---|---|---|
| H1 T1 | 0.916 ± 0.026 | 0.299 ± 0.039 | 0.070 ± 0.013 | 0.120 ± 0.024 | 0.128 ± 0.042 | 0.247 ± 0.044 | 0.978 ± 0.008 | 0.988 ± 0.003 | 0.828 ± 0.026 / 0.189 ± 0.024 |
| H1 T2 | 0.916 ± 0.026 | 0.299 ± 0.039 | 0.070 ± 0.013 | 0.120 ± 0.024 | 0.128 ± 0.042 | 0.247 ± 0.044 | 0.978 ± 0.008 | 0.988 ± 0.003 | 0.828 ± 0.026 / 0.189 ± 0.024 |
| H1 T3 | 0.946 ± 0.004 | 0.384 ± 0.027 | 0.125 ± 0.034 | 0.209 ± 0.037 | 0.197 ± 0.031 | 0.428 ± 0.105 | 0.988 ± 0.003 | 0.992 ± 0.002 | 0.882 ± 0.039 / 0.298 ± 0.076 |
| H1 T4 | 0.854 ± 0.013 | 0.225 ± 0.006 | 0.024 ± 0.008 | 0.076 ± 0.007 | 0.085 ± 0.008 | 0.110 ± 0.006 | 0.957 ± 0.008 | 0.981 ± 0.000 | 0.733 ± 0.032 / 0.103 ± 0.022 |
| A4 T1 | 0.818 ± 0.030 | 0.382 ± 0.052 | 0.157 ± 0.030 | 0.237 ± 0.062 | 0.164 ± 0.035 | 0.441 ± 0.127 | 0.985 ± 0.003 | 0.993 ± 0.002 | 0.882 ± 0.003 / 0.266 ± 0.031 |
| A4 T2 | 0.861 ± 0.003 | 0.505 ± 0.046 | 0.221 ± 0.016 | 0.418 ± 0.105 | 0.287 ± 0.035 | 0.615 ± 0.063 | 0.994 ± 0.001 | 0.996 ± 0.001 | 0.907 ± 0.012 / 0.351 ± 0.017 |
| A4 T3 | 0.849 ± 0.012 | 0.448 ± 0.042 | 0.184 ± 0.014 | 0.345 ± 0.074 | 0.228 ± 0.027 | 0.534 ± 0.075 | 0.992 ± 0.001 | 0.995 ± 0.001 | 0.899 ± 0.009 / 0.309 ± 0.017 |
| A4 T4 | 0.670 ± 0.022 | 0.208 ± 0.013 | 0.047 ± 0.010 | 0.056 ± 0.025 | 0.074 ± 0.004 | 0.128 ± 0.019 | 0.951 ± 0.005 | 0.979 ± 0.003 | 0.729 ± 0.047 / 0.113 ± 0.019 |

DEV selects T3 for all three H1 seeds and T2 for all three A4 seeds. Both choices trade deepset
recall for benign false alarms on the fresh MASSIVE-dev set (H1: 0.247 → 0.428; A4: 0.441 → 0.615).
The DEV benign groups (Dolly, OASST2, Aya, forum prompts) are separated well at low thresholds,
but short voice-assistant requests (MASSIVE) are not; DEV does not contain that kind of text, so the
DEV criterion cannot see the cost. **No threshold improves deepset recall without raising benign
false alarms: the limit is the model's ranking, not the threshold.** The iteration-3 T1 threshold is
kept as the reference operating point.

## Language and question-form generalization

- Cross-language ([json](v5_iter4_language_generalization.json)): with English-only attacks in
  TRAIN, deepset German attacks are caught at 0.10–0.14 by H1 and 0.17–0.29 by the A4 control
  (T1, per seed); English ones at 0.29–0.38 (H1) and 0.38–0.47 (A4).
  Benign German/Turkish text is handled well (MASSIVE-dev FPR de 0.16, tr 0.19 for H1 seed 42).
  Training with real German/Turkish attacks could not be run.
- Question form ([json](v5_iter4_question_generalization.json)): question-shaped attacks are caught
  in-source (DEV 0.89; HackAPrompt TEST 1.0) but not in deepset (H1 mean 0.13 at T1; 0.43 for deepset's non-question attacks, seed 42). The model learned source-specific question attacks, not the form.

## What would move this forward

1. A real, human-written Turkish/German attack source. None is public today (checked in this
   iteration). Options: collect one (e.g. a Turkish/German red-team exercise with consent and a
   clear license), or accept a second stage with explicitly flagged synthetic data.
2. If synthetic data is accepted, the rules of this iteration apply: `synthetic=true`, original
   source, transformation method and model recorded, TRAIN only, never in TEST/OOD. The best
   candidate is `3nesdeniz/turkish-conversation-prompt-injection` (CC BY 4.0, author-written, paired
   benign boundary rows), plus machine translation of existing English attacks into German.
3. Add short voice-assistant/command-style benign text to DEV (not MASSIVE) so threshold selection
   can see that cost.
