# V5 iteration 2 plan: benign distribution repair (fixed before training)

Written after the root-cause diagnostics and before any iteration-2 training. E1–E8 and all V5 v1
results stay unchanged; E6 is the immutable reference (M0). V2 stays the firewall default.

## Data

- Attack TRAIN is fixed for every run: Tensor Trust + JailbreakLLMs jailbreak rows exactly as in E6.
- Existing benign TRAIN (JailbreakLLMs regular prompts, security docs, arXiv) is kept in every run.
- New benign TRAIN: Dolly (en), OASST2 (en/de/tr), Aya (en/de/tr); see
  [benign_source_research.md](benign_source_research.md). deepset and MASSIVE never enter TRAIN/DEV.
- DEV (shared by every run): V5 DEV + new-source DEV. TEST1: V5 TEST + new-source TEST.
  TEST2: deepset, Gandalf. TEST2_OOD_BENIGN: MASSIVE en/de/tr.

## Runs (all: E6 recipe = focal gamma 2.0 alpha 0.25, 510 content tokens, AdamW 2e-5, batch 16, 3 epochs, seed 42)

To keep training time reasonable, batches are length-grouped (seeded shuffle, sort inside chunks of
50 batches, shuffled batch order). Because E6 used plain random batches, R0 retrains E6's data
with the new batching as a control.

| Run | New benign added | Sampling |
|---|---|---|
| R0 | none (E6 data) | BALANCED_SOURCE |
| B1 | English (Dolly, OASST en, Aya en) | BALANCED_SOURCE |
| B2 | German (OASST de, Aya de) | BALANCED_SOURCE |
| B3 | Turkish (Aya tr, OASST tr) | BALANCED_SOURCE |
| B4 | English + German + Turkish (= M2 balanced, = M3 multilingual) | BALANCED_SOURCE, new sources 1x |
| M1 | English + German + Turkish | NATURAL |
| A2 | English + German + Turkish | BALANCED_SOURCE, new benign sources weighted 2x |
| A4 | English + German + Turkish | BALANCED_SOURCE, new benign sources weighted 4x |

M4/M5 (short-instruction hard negatives) are not separate runs: 94.5% of the new benign TRAIN
rows are already ≤94 tokens (11,143 of 11,791), so a "short-only" subset would duplicate B1/B4.
Length effects are measured by the length-bucket reports instead.

## Selection (DEV only)

- Threshold per run: maximize DEV macro accuracy over (source, label) groups present in DEV.
- Temperature per run: fitted on DEV.
- REGRESSION = DEV Tensor Trust recall or DEV JailbreakLLMs recall more than 0.05 below E6 on the
  same DEV at E6's frozen threshold.
- Candidate: among non-regression runs, lowest mean benign FPR over DEV benign sources; ties by
  mean attack recall, then DEV PR-AUC, then DEV ECE. Raw metrics for every run are reported.

## Status rules (for the one-shot TEST)

A deepset FPR ≤ 0.40 and at least 0.25 below E6 (0.707). B MASSIVE FPR ≤ 0.15 overall and ≤ 0.25 in
each language, and below E6's MASSIVE FPR. C Tensor Trust recall ≥ 0.95. D JailbreakLLMs recall ≥
0.70. E security docs and arXiv FPR ≤ 0.10. F calibrated TEST1 ECE ≤ 0.05. G = B (improvement not
specific to deepset).

- REGRESSION: C or D fails.
- READY_FOR_NEW_EXTERNAL_VALIDATION: A–G all hold.
- GENERALIZATION_IMPROVED: B–F hold, deepset improves by ≥ 0.15 but A fails.
- PARTIAL: deepset or MASSIVE FPR improves by ≥ 0.10 vs E6, other criteria not met.
- NO_GO: otherwise.
- GENERALIZATION FAILURE flag: deepset improves but MASSIVE FPR is worse than E6's.
- PRODUCTION_READY is never an outcome.
