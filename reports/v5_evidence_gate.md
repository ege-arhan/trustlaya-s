# V5 evidence gate

**V5_STATUS = GO**

GO means the evidence required *before* training exists. It does not mean V5 is trained, better than V2, or deployable. No training was started by this gate.

| Check | Result | Evidence |
|---|---|---|
| Tensor Trust parser correct | PASS | v1 raw dump reproduces upstream candidates for 490/492 hijacking and 438/438 extraction benchmark attacks; schema surprises raise SchemaError (tests/test_v5_dataset.py) |
| Tensor Trust source semantics preserved | PASS | only upstream success heuristics/benchmarks map to ATTACK; failed attempts, self attacks, access codes and defenses are EXCLUDED, never BENIGN |
| checksum reproducible | PASS | pinned SHA-256 for every source file; manifest/lineage/mapping checksums match; rebuild not run in this invocation |
| train/test leakage none | PASS | 21833 clusters each in one split; 0 train/dev-test cluster overlaps across experiments A-D |
| cross-source dedup done | PASS | MinHash LSH, char 5-grams, 128 permutations, verified Jaccard > 0.7; Tensor Trust <-> JailbreakLLMs overlap rows: 49 |
| context ablation done | PASS | HEAD/TAIL/HEAD_TAIL at 94/128/256/510 + sliding windows on the same V2 checkpoint; 512 is not a valid content length |
| tokenizer audit done | PASS | English-dominant Tensor Trust/JailbreakLLMs vs native Turkish text |
| V2 public baseline done | PASS | frozen V2 at native HEAD-94 on DEV, TEST and OOD_TEST per source, fixed threshold 0.5 |
| calibration audit done | PASS | temperature fitted on DEV only; per-source ECE/Brier/precision/recall/FPR |

## Warnings (not gate items)

- Tensor Trust data repository has no LICENSE file (game code is BSD-2-Clause); status reported as-is. Raw text is not committed or redistributed.
- HackAPrompt not included: gated dataset, terms not yet accepted by the account owner.
- REAL_V2_REDACT_E2E = VERIFIED (1/4 PII fixtures); 12/20 firewall fixtures matched expectations.
- Security-prose hard negatives were filtered by rules (code/quotes/imperative payload lines removed), not reviewed by people.
- V2 context ablation reuses weights trained at 94 tokens; it measures reading strategy under distribution shift, not a 512-token model.
- DEV temperature fit reached the search bound (20.0): V2 attack scores carry little ranking signal on these sources, so scaling mostly flattens them.
