# V5 iteration 5 plan: controlled synthetic multilingual augmentation (fixed before training)

Synthetic data is **TRAIN AUGMENTATION ONLY**. It is not gold, not human-labeled and never enters
DEV, TEST1, TEST2, deepset or MASSIVE. Provenance for every synthetic row:
`data/v5_iter5_manifest.json` (synthetic flag, method, original source and id, source/target
language, model and revision, timestamp, source checksum, question form).

## Synthetic sources

- `tr_curated`: 3nesdeniz/turkish-conversation-prompt-injection (CC BY 4.0), 150 attacks + 600
  benign rows; the card states the text is synthetic (author-curated).
- `mt_de_attack` / `mt_tr_attack`: OPUS-MT (Helsinki-NLP, CC BY 4.0) translations of 1,100 / 1,000
  English TRAIN attacks (≤ 96 words, equal per source: Tensor Trust, JailbreakLLMs, HackAPrompt).
- `mt_de_benign` / `mt_tr_benign`: translations of 1,100 / 1,000 short Dolly TRAIN questions, to
  control for a "machine-translated text = attack" shortcut.
- Any synthetic row that clusters (MinHash, Jaccard > 0.70) with a TEST/OOD row is dropped.

## Runs (H1 recipe: A4 data + HackAPrompt, focal, 510 tokens, grouped batches, 3 epochs)

| Run | Adds to S0 | Seed |
|---|---|---|
| S0 | nothing (= iteration-3 H1_grouped, reused, seeds 42/43/44) | 42, 43, 44 |
| S1 | Turkish synthetic attacks (tr_curated + mt_tr_attack) | 42 |
| S2 | German synthetic attacks (mt_de_attack) | 42 |
| S3 | S1 + S2 attacks | 42 |
| S4 | S3 + synthetic benign (tr_curated benign, mt_de_benign, mt_tr_benign) | 42 |
| S5 | S2 + mt_de_benign (short German benign questions) | 42 |

Synthetic attacks stay ≤ 15% of attack TRAIN (S3 ≈ 14.7%). Their sampling mass equals their count
share within their label, so source balancing cannot inflate them. The 25% variant is not run in
this iteration (≤ 15% first, as instructed).

## Selection (DEV only; the iteration-3 DEV contains no synthetic rows)

- Threshold: iteration-3 class-balanced rule (T1). Temperature fitted on DEV.
- The two runs among S1–S5 with the highest DEV class-balanced accuracy (tie: DEV PR-AUC) get
  seeds 43 and 44. Thresholds, temperatures and checkpoints are locked before one TEST run.
- DEV has no German/Turkish attacks (synthetic DEV is not allowed), so DEV cannot see the target
  effect directly; this is a known limit of the design.

## Status rules (candidate = best of the two 3-seed configs by DEV; compared with S0's 3-seed mean)

- A deepset recall ≥ 0.40 (S0 0.299). B deepset FPR ≤ 0.15. C fresh MASSIVE-dev FPR ≤ S0 + 0.03 and
  MASSIVE-test FPR ≤ S0 + 0.03. D Tensor Trust recall ≥ 0.95. E JailbreakLLMs recall ≥ 0.75.
  G sd of deepset recall over seeds ≤ 0.05.
- F (an independent, non-deepset multilingual attack source improves) **cannot be assessed**: no
  such real source exists (iteration 4). Therefore GENERALIZATION_IMPROVED and
  READY_FOR_EXTERNAL_VALIDATION are not reachable in this iteration.
- PARTIAL (a benchmark-specific effect): A, B, C, D, E and G hold. NO_GO: otherwise.
- PRODUCTION_READY is never an outcome. V2 stays the default and firewall model.
