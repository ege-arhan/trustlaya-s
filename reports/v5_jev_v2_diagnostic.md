# Frozen V2 on the Jev SILVER corpus

**Date:** 2026-09-25. **Status:** diagnostic only. The [machine-readable result](../benchmarks/v5/jev_v2_diagnostic.json) contains all source counts, ranking diagnostics, a 0.00–1.00 threshold curve, latency and artifact hashes. Row predictions are local and gitignored. **Jev SILVER is not human GOLD. The numbers below are agreement with named references, not independent model accuracy.**

## Protocol

- Input: exactly 600 rows from the versioned [Jev SILVER run](v5_jev_only_silver.md); each text hash checked. Jev returned `ATTACK`, `BENIGN_DUAL_USE`, `NORMAL`, or `UNRESOLVED` before this evaluation.
- Model: frozen TrustLaya-S **V2** and its frozen FP32 ONNX, both checked against the SHA-256 values in the [V2 freeze](v2_external_baseline_frozen.md). ONNX CPU; raw `prompt_injection` head; threshold **0.50** fixed by the prior external baseline. The three Jev `UNRESOLVED` rows are excluded from binary agreement.
- V2's actual `Analyzer` uses a **96-token maximum input**, giving about **94 content tokens**. The tokenizer architecture's 512-token maximum is not the production reading limit. Text lengths were counted after V2 normalization.
- No model parameters, policy thresholds, calibration objects or default gateway configuration were changed. No threshold was selected from the SILVER labels.

## Measured V2 output

V2 flagged **364/600** texts at its frozen 0.50 raw-score threshold. Against the Jev SILVER projection on the other 597 rows:

| Reference | N | Reference attacks | V2 TP | V2 FP | V2 TN | V2 FN | Precision vs reference | Recall vs reference | F1 vs reference | V2 flag rate on reference negatives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jev SILVER | 597 | 95 | 90 | 273 | 229 | 5 | 0.248 | 0.947 | 0.393 | 0.544 |
| deepset **train** source proxy, different task projection | 340 | 170 | 147 | 61 | 109 | 23 | 0.707 | 0.865 | 0.778 | 0.359 |

The Jev SILVER ranking diagnostic is ROC-AUC **0.750**, average precision **0.382**. Against the deepset train *source proxy*, the corresponding figures are **0.810** and **0.796**. The references disagree on 84/340 rows (including three unresolved), and the deepset train partition is not an independent hidden test. Consequently none of these figures can replace the published external V2 benchmark.

By source, V2 marked **36/79 OWASP**, **44/92 Microsoft**, **76/89 Stack Exchange**, and **208/340 deepset train** rows above 0.50. The first three are security-documentation/question candidate sources, not independently annotated benign gold; these counts are *attack-flag rates*, not false-positive rates. The recorded policy actions include BLOCK and REVIEW on documents because `Analyzer` was invoked as if each text were a request. Those actions are not measured agent-tool enforcement outcomes.

## Threshold and context diagnostics

Thresholds in the file are an **offline sweep only**; the deployed 0.50 threshold remains unchanged.

| Raw threshold | Jev attacks flagged / 95 | Jev negatives flagged / 502 | F1 vs Jev |
|---:|---:|---:|---:|
| 0.50 | 90 | 273 | 0.393 |
| 0.80 | 68 | 197 | 0.378 |
| 0.95 | 39 | 78 | 0.368 |

At the real 94-content-token reading boundary, **205/600** inputs are truncated. For the 205 long inputs, V2 flags **124/192** Jev negatives and captures **12/13** Jev attacks. For the 392 non-truncated, resolved inputs, it flags **149/310** Jev negatives and captures **78/82** Jev attacks. Source and length are confounded here: many long texts are security documentation. These numbers do not isolate truncation as the cause, but the long-input disagreement is worse. Only 32 rows exceed the tokenizer's 510-content-token architecture window, and only two of those are Jev `ATTACK`; this set is insufficient to validate long-attack handling.

The recorded V2 project train split contains **6,972** examples. Against this split, the 600-row corpus has zero normalized exact overlaps, zero character 4–5-gram cosine similarities ≥0.85, and maximum similarity 0.473. This check does not cover every data source or encoder pretraining, so it is not a universal contamination guarantee.

V2 ONNX CPU model inference on this Mac: p50 **4.44 ms**, p95 **4.73 ms**; total `Analyzer` call: p50 **4.76 ms**, p95 **5.59 ms**. These are this run's batch=1 observations, not UNO Q measurements. Jev/Vercel call latency is documented separately in the [SILVER run](v5_jev_only_silver.md).

## Training decision

**No V5 training from this corpus.** Jev failed to mark 82 of deepset's 170 source-label-1 examples as `ATTACK`, while V2 still flagged 273 Jev-negative rows. Training directly on these SILVER targets risks teaching a new head to suppress real attacks or to imitate Jev's task interpretation. Raising the production threshold would sharply reduce attack coverage even relative to Jev. The corpus also lacks an untouched, source-separated and sufficiently long evaluation set. The next defensible step is to acquire a new license-clear independent attack/discussion evaluation source, define matching label semantics, and test a small candidate only against a threshold frozen before that evaluation. If human labeling remains excluded, label agreement must continue to be reported as proxy agreement.
