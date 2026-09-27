# V5 iteration 3 plan: short human attacks (HackAPrompt), fixed before training

Written after iteration 2 and the start of the variance study, before any iteration-3 training.

## Why

Iteration 2 cut benign false alarms but the DEV-selected model caught only 21% of deepset attacks.
Two causes are addressed here, each disclosed as informed by iteration-2 results:
1. TRAIN had no short, question-like human attacks (Tensor Trust is game-specific; JailbreakLLMs
   is long). HackAPrompt successful inputs are short (median 18 words) and human-written.
2. The iteration-2 threshold rule averaged accuracy over 8 DEV groups, 6 of them benign, which
   pushes the threshold up. Iteration 3 uses a **class-balanced** rule: mean accuracy of attack
   groups and mean accuracy of benign groups weighted equally.

## Data (see reports/iter3_dataset.json)

- Attack TRAIN: Tensor Trust + JailbreakLLMs (as E6) + HackAPrompt successful inputs (5,000; DEV 1,000;
  TEST 1,000). Failed HackAPrompt attempts are excluded, never benign. 71% of HackAPrompt rows
  contain "pwn" (the competition's target word); recall is reported separately with and without it.
- Benign TRAIN: as A4 (V5 v1 benign + Dolly/OASST2/Aya en/de/tr, new benign sources weighted 4x).
- Tests: TEST1 (pool TEST incl. HackAPrompt TEST), TEST2 deepset + Gandalf, MASSIVE test (all
  three already scored in earlier iterations, so they are reused, not blind), and
  **MASSIVE dev partition as a fresh OOD benign test never scored before** (2,959 rows).

## Runs (focal gamma 2.0 alpha 0.25, 510 tokens, AdamW 2e-5, batch 16, 3 epochs)

| Run | Data | Batching | Seed |
|---|---|---|---|
| H1_grouped_s42/s43/s44 | A4 data + HackAPrompt | length-grouped | 42, 43, 44 |
| H1_random_s42 | A4 data + HackAPrompt | random (as E6) | 42 |
| control | A4 data, no HackAPrompt | length-grouped | 42, 43, 44 (variance-study runs) |

## Selection and evaluation

- Per run: class-balanced DEV threshold (grid 0.01–0.99) and DEV temperature; frozen in a lock file
  before any TEST scoring. No run is selected on TEST. The configuration verdict uses the mean over
  the three H1_grouped seeds (sd reported).
- Status rules (config mean on TEST): deepset recall ≥ 0.60 and deepset FPR ≤ 0.30; fresh MASSIVE-dev
  FPR ≤ 0.20 overall and ≤ 0.30 per language; Tensor Trust recall ≥ 0.95; JailbreakLLMs recall ≥ 0.70;
  HackAPrompt TEST recall ≥ 0.90 overall and ≥ 0.80 on rows without "pwn"; docs/arXiv FPR ≤ 0.10;
  calibrated TEST1 ECE ≤ 0.05.
  - REGRESSION: Tensor Trust < 0.95, JailbreakLLMs < 0.70 or deepset recall < 0.50.
  - READY_FOR_NEW_EXTERNAL_VALIDATION: every rule holds.
  - GENERALIZATION_IMPROVED: all attack rules hold and at most one benign-FPR rule fails.
  - PARTIAL: deepset recall or fresh MASSIVE FPR improves over the A4 control by ≥ 0.10, other rules not met.
  - NO_GO: otherwise. PRODUCTION_READY is never an outcome. V2 stays the default either way.
