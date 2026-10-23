# V5 pre-training diagnostic verdict: benchmark in preparation

**Decision:** stop before V5 training. V2 stays frozen and default; V3/V4 remain NO-GO, unpublished research checkpoints. No V5 model or human-validated benchmark score exists.

## What was completed

- Froze exact v2/v3/v4 comparison predictions, model/tokenizer checksums, thresholds, calibration state and dataset revisions. The read-only local snapshot is verified by `scripts/freeze_v5_baselines.py`; the tracked [manifest](../benchmarks/v5/frozen_manifest.json) identifies every file.
- Acquired 1,606 human-origin **candidate** rows from four source families. Removed 438 normalized exact duplicates, leaving 1,168 unique candidate rows. Tensor Trust data license is unspecified, so its raw data remain private and are not approved for redistribution/training.
- Prepared a deterministic, source-blind **600-row** packet for two human reviewers and a separate senior adjudicator. Sampling balances game-candidate length groups and uses frozen-head disagreement for coverage. Gold labels: **0**; disagreement: **not measured**; unresolved intent is not a learnable class.
- Tokenized every candidate: 612/1,168 (52.4%) exceed the 94-content-token current reading limit. Attack position is not known without review.
- Screened prior v2/v4 train/dev and historical JLL by normalized hash and character n-gram cosine ≥0.85. One v4 DEV near match was excluded from the review packet; no near match was found against v2 train, v4 train or historical JLL at that threshold.
- Measured eight read strategies on the **old V4 development proxy only**, using unchanged v2/v3/v4 weights and fixed old thresholds. V4 max pooling gave recall 0.703 / FPR 0.257 with 5.20 windows/input; the simple router gave recall 0.568 / FPR 0.219 with 2.12 windows/input. The proxy is not independent and is heavily attack-enriched, so this does not select a winner.
- After the partial Council review, added hierarchical intent annotation, a third adjudicator role, Level 1 Krippendorff alpha, and a no-training 94-versus-510-token ablation. On the same old proxy, v4 native first-510 recall rose from 0.432 to 0.626 while FPR rose from 0.181 to 0.210 at the unchanged old threshold. This is diagnostic, not a new validated operating point. A pinned tokenizer comparison found more subwords per English-oriented word for the current Turkish tokenizer; no tokenizer or model was replaced. See [Council evidence audit](v5_council_review.md), [native context](v5_native_context_diagnostic.md), and [tokenizer audit](v5_tokenizer_audit.md).

## What the diagnostic can and cannot answer

**Truncation:** 94-token reading discards text in 52.4% of the new unlabeled candidates and 94.5% of historical JLL attacks. This establishes exposure loss, not the causal fraction of classifier errors. Gold attack location is pending.

**Attack vs discussion:** old source-derived labels confound game attempts with security documents. No human-reviewed matched intent pairs exist yet, so the contribution of discussion confusion cannot be quantified.

**Threshold/calibration:** old fits are incompatible with assuming calibration after new aggregation. V4's JLL calibration failed to transfer. No new threshold or calibration was selected.

**Read strategy:** on the old proxy, max pooling has the highest V4 recall/F1 among the eight measured; it also has the highest FPR and more model windows. The router spends fewer windows with lower recall. Source-separated human DEV is required for selection.

**Cross-source generalization:** not measured for V5. The four acquired source families lack comparable reviewed attack/benign distributions, and two untouched hidden sources are not frozen. No hidden-test tuning occurred.

## Why V3 and V4 failed to become default

V3 reduced historical JLL FPR to 0.052 but detected only **22/635** attacks (recall 0.035, F1 0.048). V4 restored **572/635** attack detections (recall 0.901) but falsely flagged **3,370/5,126** benign JLL items (FPR 0.657), including FPR 0.860 above 510 content tokens. It detected **4/60** positives on the separate deepset prompt-injection task. Thus neither provided an acceptable recall/false-alarm/generalization balance. The old JLL source had been previously examined and is not a fresh V5 blind test.

## Exact evidence needed before V5 could replace V2

1. At least 600 independently double-annotated primary examples, a genuinely separate senior adjudicator, Level 1 nominal alpha target ≥0.80, explicit unresolved exclusions, per-class coverage and human attack-location labels for long attacks. A 600-row packet alone does not meet this gate.
2. Legally usable, provenance-pinned TRAIN and DEV plus two source-separated, untouched hidden tests with enough attacks **and** benign security discussions in each; no synthetic primary rows, no source/class shortcut, and a valid Turkish subset or an explicit absence statement.
3. Exact/normalized/near-duplicate and semantic overlap audit against v2/v3/v4 training and between all new splits, with contaminated rows excluded from headline scores.
4. On reviewed DEV, a fixed context strategy, threshold and per-strategy calibration selected before hidden tests; complete threshold curves, ECE/Brier/NLL and latency/call counts.
5. On both hidden sources at the frozen operating point, materially better attack F1 and benign FPR than v2 **without recall collapse**, including long-context and quoted/discussion slices; paired row predictions, uncertainty intervals where sample size supports them, and at least 100 FP/FN case reviews where available.
6. Regression tests and gateway behavior unchanged, plus deployment resource/latency measurement on target hardware before any UNO Q claim.

These conditions are currently unmet. This report is a research checkpoint, not a V5 release or production-readiness claim.
