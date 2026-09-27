# V5 iteration 2: benign distribution repair (8 runs)

**V5_STATUS = PARTIAL** (pre-registered rules). DEV-selected candidate: `A4_multilingual_4x`
(checkpoint `69995a93efbe9a207d13d9a57031a793cfe9dee0362fa76501341e670cd08ab1`). **V2 remains the default and the
firewall model; no V5 weights are published.** E1–E8 and all V5 v1 results are unchanged.

Plan and rules were written before training ([benign_repair_plan.md](benign_repair_plan.md));
selections, thresholds and temperatures were frozen in
[`v5_benign_repair_lock.json`](v5_benign_repair_lock.json) before TEST1, TEST2 and the new OOD
benign test were scored once. deepset and MASSIVE never entered TRAIN or DEV; no threshold,
temperature or model choice used them. Per-run files: `reports/experiments/benign_repair/<run>/`.
Master: [`v5_benign_repair_master.json`](v5_benign_repair_master.json).

## Two findings the status rules do not capture

1. **deepset attack recall collapsed.** A4 catches 21% of deepset attacks (E6 71%, V2 87%). Its
   deepset FPR fell from 0.707 to 0.058, but largely because its DEV-chosen threshold is too high
   for deepset-style attacks: deepset PR-AUC actually rose (E6 0.510 → A4 0.634). The rules only
   protected Tensor Trust and JailbreakLLMs recall, so this is not labeled REGRESSION, but it is one.
2. **Batching alone changed the model a lot.** R0 retrains E6's exact data with length-grouped
   batches (the only change). R0 lost most JailbreakLLMs recall (0.779 → 0.331) and deepset recall
   (0.714 → 0.250) and is itself a DEV regression. So the B/M/A runs must be compared with R0, not
   E6, and every run is a single seed; run-to-run variance is unmeasured and may be large.

## Final metrics (TEST1, TEST2, TEST2_OOD_BENIGN; each run at its frozen DEV threshold)

MASSIVE FPR is shown as overall (en / de / tr). ECE is TEST1 raw → DEV-temperature calibrated.

| Run | TT recall | JLL recall / FPR | Gandalf recall | deepset FPR | deepset recall | deepset PR-AUC | MASSIVE FPR (en / de / tr) | docs / arXiv FPR | Dolly / OASST / Aya FPR | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| M0_E6_reference | 0.991 | 0.779 / 0.115 | 0.994 | 0.707 | 0.714 | 0.510 | 0.944 (0.855 / 0.979 / 0.998) | 0.000 / 0.011 | 0.502 / 0.381 / 0.694 | 0.126 → 0.106 |
| R0_control | 0.910 | 0.331 / 0.010 | 0.953 | 0.149 | 0.250 | 0.509 | 0.294 (0.229 / 0.325 / 0.329) | 0.000 / 0.000 | 0.065 / 0.077 / 0.091 | 0.119 → 0.111 |
| B1_en | 0.971 | 0.628 / 0.054 | 0.938 | 0.048 | 0.230 | 0.618 | 0.367 (0.170 / 0.388 / 0.544) | 0.000 / 0.011 | 0.005 / 0.023 / 0.035 | 0.062 → 0.027 |
| B2_de | 0.931 | 0.471 / 0.022 | 0.932 | 0.048 | 0.171 | 0.532 | 0.270 (0.450 / 0.025 / 0.337) | 0.000 / 0.000 | 0.072 / 0.077 / 0.040 | 0.127 → 0.113 |
| B3_tr | 0.963 | 0.430 / 0.028 | 0.961 | 0.182 | 0.329 | 0.506 | 0.347 (0.521 / 0.503 / 0.018) | 0.000 / 0.011 | 0.123 / 0.147 / 0.054 | 0.087 → 0.075 |
| B4_multilingual | 0.988 | 0.767 / 0.112 | 0.982 | 0.091 | 0.282 | 0.620 | 0.293 (0.401 / 0.234 / 0.244) | 0.000 / 0.022 | 0.015 / 0.055 / 0.015 | 0.038 → 0.013 |
| M1_multilingual_natural | 0.989 | 0.657 / 0.064 | 0.983 | 0.086 | 0.314 | 0.691 | 0.272 (0.402 / 0.211 / 0.204) | 0.071 / 0.033 | 0.013 / 0.051 / 0.012 | 0.048 → 0.015 |
| A2_multilingual_2x | 0.986 | 0.779 / 0.122 | 0.979 | 0.086 | 0.262 | 0.636 | 0.281 (0.414 / 0.189 / 0.242) | 0.000 / 0.011 | 0.017 / 0.042 / 0.012 | 0.042 → 0.013 |
| A4_multilingual_4x | 0.982 | 0.727 / 0.120 | 0.964 | 0.058 | 0.214 | 0.634 | 0.198 (0.291 / 0.103 / 0.201) | 0.010 / 0.011 | 0.005 / 0.035 / 0.010 | 0.049 → 0.017 |
| V2 (reference, v1 test) | 0.443 | 0.901 / 0.875 | 0.722 | 0.364 | 0.865 | – | not measured | 0.898 / 0.978 | not measured | – |

## DEV selection (the only basis for choosing)

| Run | Threshold | Temperature | DEV TT recall | DEV JLL recall | Mean DEV benign FPR | DEV PR-AUC | REGRESSION (vs E6 DEV) | Train min | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| R0_control | 0.73 | 0.8802 | 0.889 | 0.333 | 0.038 | 0.921 | yes | 18.3 | `c116ca7735fa9fe4…` |
| B1_en | 0.54 | 0.6231 | 0.944 | 0.610 | 0.018 | 0.959 | yes | 20.7 | `949bb73b89193a6a…` |
| B2_de | 0.75 | 0.8043 | 0.890 | 0.452 | 0.031 | 0.926 | yes | 15.3 | `6590cc8ad41de8da…` |
| B3_tr | 0.71 | 0.8541 | 0.923 | 0.475 | 0.051 | 0.935 | yes | 16.9 | `3a52dc36b770969c…` |
| B4_multilingual | 0.32 | 0.7027 | 0.971 | 0.729 | 0.031 | 0.970 | no | 23.2 | `6cbfce383ab312a3…` |
| M1_multilingual_natural | 0.25 | 0.5957 | 0.976 | 0.667 | 0.032 | 0.977 | yes | 20.1 | `1e757730361e5d71…` |
| A2_multilingual_2x | 0.34 | 0.6922 | 0.967 | 0.757 | 0.035 | 0.970 | no | 23.3 | `dbf5df55fde54ca2…` |
| A4_multilingual_4x | 0.41 | 0.6819 | 0.963 | 0.751 | 0.029 | 0.968 | no | 21.4 | `69995a93efbe9a20…` |

E6 on the same DEV at its frozen threshold: Tensor Trust recall 0.981, JailbreakLLMs recall 0.768.
REGRESSION = a DEV attack recall more than 0.05 below that.

## Answers

1. **New public benign sources:** Dolly 15k, OpenAssistant oasst2, Aya dataset (TRAIN/DEV/TEST);
   MASSIVE 1.1 (OOD benign test only). Details: [benign_source_research.md](benign_source_research.md).
2. **License/provenance:** Dolly CC BY-SA 3.0 (Databricks employees); oasst2 Apache-2.0 (volunteers,
   reviewed, not spam); Aya Apache-2.0 (annotators); MASSIVE CC BY 4.0 (en crowd-written, de/tr
   professional localizations). All pinned by revision and SHA-256.
3. **Languages:** English, German, Turkish. German is small (544 TRAIN rows).
4. **Samples:** new benign TRAIN 11,791 (en 8,382 / de 544 / tr 2,865), DEV 2,420, TEST 2,413;
   MASSIVE OOD benign 2,978 (990 en / 996 de / 992 tr). Full counts: `benign_repair_dataset.json`.
5. **Hard negatives with security vocabulary (new TRAIN):** 246, plus 503 security-doc and 392
   arXiv rows from V5 v1.
6. **Short instructions (≤94 tokens, new TRAIN):** 11,143 of 11,791.
7. **Security-vocabulary rows:** see `security_vocabulary_rows` in `benign_repair_dataset.json`.
   Root-cause analysis found no keyword shortcut in E6 ([benign_root_cause.md](benign_root_cause.md)).
8. **E6 vs runs:** table above. Against the R0 control, adding multilingual benign text (B4)
   restored JailbreakLLMs recall (0.331 → 0.767), cut deepset FPR (0.149 → 0.091) and new-source
   FPR (≈0.08 → 0.015–0.055), and left MASSIVE FPR flat (0.294 → 0.293).
9. **Language FPR (MASSIVE):** each language-specific run fixes mainly its own language: B1 en 0.170,
   B2 de 0.025, B3 tr 0.018, while the other two languages stay at 0.34–0.54. Mixing all three (B4)
   gives 0.40 / 0.23 / 0.24; 4x weight (A4) gives 0.29 / 0.10 / 0.20. This is direct evidence for H1.
10. **Length FPR:** MASSIVE is almost entirely ≤32 tokens, so its FPR is the short-input FPR.
    Per-cell (language × length) FPR, recall, F1 and mean score for every source and run are in
    the master file under `details`.
11. **deepset FPR:** E6 0.707 → A4 0.058 (R0 0.149). See finding 1: recall fell to 0.214.
12. **New OOD benign (MASSIVE) FPR:** E6 0.944 → A4 0.198 (R0 0.294). Not below 0.15; English
    voice commands stay hardest (0.29).
13. **Tensor Trust recall:** 0.98–0.99 for all multilingual runs (A4 0.982).
14. **JailbreakLLMs:** A4 recall 0.727, FPR 0.120 (E6 0.779 / 0.115).
15. **Calibration:** A4 TEST1 ECE 0.049 raw → 0.017 with the DEV temperature (0.68). E6 on the new
    TEST1 is 0.106 calibrated, because it misfires on the new benign sources.
16. **Training time:** 15–23 minutes per run (Apple MPS, fp32, 3 epochs, length-grouped batches).
17. **Model size:** 42.13M parameters, 168.5 MB fp32 safetensors (same as E6).
18. **Candidate checksum:** `69995a93efbe9a207d13d9a57031a793cfe9dee0362fa76501341e670cd08ab1` (all runs in the DEV table).
19. **V5_STATUS:** PARTIAL. Criteria: A deepset FPR ✓, B MASSIVE ✗ (0.198 > 0.15; en 0.29 > 0.25),
    C Tensor Trust ✓, D JailbreakLLMs ✓ (0.727 ≥ 0.70), E docs/arXiv ✓, F calibration ✓,
    G not deepset-only ✗. Plus the unregistered deepset recall regression above.
20. **V2 default:** yes. V2 remains the default and the firewall model; the real-V2 firewall results
    are untouched.

## What this shows and what it does not

- H1 (language) is confirmed causally: adding benign text in a language removes most false alarms
  in that language on an independent source (MASSIVE).
- More benign weight (1x → 2x → 4x) lowers MASSIVE FPR (0.293 → 0.281 → 0.198) with small attack
  recall cost, but no run reaches the 0.15 target.
- The DEV threshold is set where Tensor Trust/JailbreakLLMs attacks and the DEV benign sources
  separate. deepset attacks are short and question-like and fall below it. Better OOD ranking (PR-AUC)
  did not translate into recall at that threshold.
- Single seed and a batching change: effect sizes between runs are not yet reliable.

## Next steps (not run)

1. Re-run R0/B4/A4 with 3 seeds and E6's original random batching to separate data effects from
   batching and seed noise.
2. Add an attack source resembling short, question-like injections to TRAIN (independent of
   deepset), then check deepset recall again.
3. Security Stack Exchange questions as security-vocabulary hard negatives.
