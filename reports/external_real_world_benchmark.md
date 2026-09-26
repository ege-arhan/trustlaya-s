# TRUSTLAYA-S EXTERNAL REAL-WORLD EVALUATION

| Dataset | Provenance | N | Language | Task | TrustLaya-S Precision | TrustLaya-S Recall | TrustLaya-S F1 | FPR | FNR | ROC-AUC / PR-AUC |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|
| TAB official test | Real public ECHR legal cases with anonymization gold | 2079 windows / 127 cases | English | Narrow DIRECT PERSON/CODE window PII | 0.120 | 0.096 | 0.107 | 0.083 | 0.904 | 0.593 / 0.125 |
| JailbreakLLMs community subset | Human/community collected prompts, author labeled | 5888 prompts | Not annotated per row | Jailbreak proxy via prompt-injection head | 0.113 | 0.945 | 0.202 | 0.900 | 0.055 | 0.707 / 0.249 |

**Interpretation:** These are separate tasks and must not be averaged. The main evaluation uses external datasets absent from the recorded TrustLaya-S training text after the documented exact/near-duplicate screen. Internal synthetic project tests are regression/stress tests only. No clinical PHI or real-secret F1 is claimed.

## Frozen setup

- v2 weight SHA-256: `99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c`; ONNX CPU; deployed tokenizer and 96-token model input; fixed raw-score threshold `0.5`. Model weights and thresholds were not changed.
- Long TAB cases were split into nonoverlapping 94-wordpiece windows with two special tokens. Gold span offsets were checked against source text. A boundary-crossing identifier drops the window. Only one quality-checked annotator was used per case. This window-level task is different from document-level PII detection.
- TAB positive means a fully contained gold `DIRECT` `PERSON` or `CODE` mention. Other `QUASI` identifiers are outside this narrow target; predicted detections on them count as false positives in this metric. The classification head is broader than this target, so this is a domain/task-transfer diagnostic, not an exhaustive PII capability estimate.
- JailbreakLLMs excludes `open_source` repository prompts; `reddit`, `discord`, and `website` are retained. A jailbreak request and a malicious instruction embedded in tool output are different tasks. The head was evaluated as a proxy, not advertised as a purpose-built jailbreak classifier.
- Both model heads truncate each evaluated window/prompt at 96 model tokens. Especially for long jailbreak prompts, this can hide decisive later content. The jailbreak BERT baseline reads up to 512 tokens, so the model comparison also reflects different input budgets.
- The deployed Turkish-first normalization is applied unchanged to English text, including mapping capital `I` to dotless `ı`. This is part of the frozen pipeline and may contribute to English domain shift.
- TAB source commit `558e09e26d6b36f5f78440074e6a233946d98bd9`; JailbreakLLMs source commit `2dbd7bbc25f1b156552678f451bddbc787cd679f`. Source repositories and licenses are in the [dataset audit](external_benchmark_dataset_audit.md).

## Sampling uncertainty

TAB F1 95% cluster-bootstrap interval across 127 source cases: 0.064–0.147. Jailbreak F1 95% row-bootstrap interval: 0.188–0.216. Each uses 1,000 seeded resamples; the jailbreak interval does not account for source-family clustering and may be optimistic.

## Contamination and exclusions

Training comparison used `data/splits/train.jsonl` and the pinned Turkish PII training source (12,082 total texts). Normalized exact matching and char 4–5-gram cosine similarity ≥0.85 were screened. No training exact/near matches were found among evaluated rows. This is a text-overlap screen; it cannot exclude all semantic contamination or unrecorded encoder pretraining.

| Dataset | Candidates | Conflicting-label rows removed | Within-benchmark duplicates removed | Training exact removed | Training near removed | Retained |
|---|---:|---:|---:|---:|---:|---:|
| TAB | 2088 | 0 | 9 | 0 | 0 | 2079 |
| JailbreakLLMs | 6174 | 24 | 262 | 0 | 0 | 5888 |

TAB also dropped 4 boundary-crossing windows before duplicate screening. JailbreakLLMs excluded 213 `open_source` rows before duplicate screening. Identical text with conflicting gold labels was entirely removed; keeping one label arbitrarily would bias the result.

## Source breakdown: community jailbreak prompts

| Source | N | Positive | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| discord | 707 | 280 | 0.405 | 0.925 | 0.564 | 0.890 | 0.075 |
| reddit | 484 | 202 | 0.423 | 0.950 | 0.585 | 0.929 | 0.050 |
| website | 4697 | 154 | 0.035 | 0.974 | 0.068 | 0.899 | 0.026 |

## Span evidence, separate from the classifier

The existing pattern-assisted extractor exactly matched 0 of 351 retained gold direct PERSON/CODE spans (exact-span recall 0.000). This is not a token-level NER model. A policy decision cannot be interpreted as successful entity extraction.

## Fixed-threshold sweep (diagnostic; no test-set selection)

| Dataset | Threshold | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| TAB | 0.10 | 0.104 | 0.178 | 0.131 | 0.181 | 0.822 |
| TAB | 0.25 | 0.119 | 0.137 | 0.127 | 0.119 | 0.863 |
| TAB | 0.50 | 0.120 | 0.096 | 0.107 | 0.083 | 0.904 |
| TAB | 0.75 | 0.132 | 0.068 | 0.090 | 0.053 | 0.932 |
| TAB | 0.90 | 0.114 | 0.041 | 0.060 | 0.038 | 0.959 |
| JailbreakLLMs | 0.10 | 0.110 | 0.984 | 0.199 | 0.960 | 0.016 |
| JailbreakLLMs | 0.25 | 0.111 | 0.965 | 0.198 | 0.940 | 0.035 |
| JailbreakLLMs | 0.50 | 0.113 | 0.945 | 0.202 | 0.900 | 0.055 |
| JailbreakLLMs | 0.75 | 0.119 | 0.915 | 0.210 | 0.824 | 0.085 |
| JailbreakLLMs | 0.90 | 0.141 | 0.862 | 0.243 | 0.634 | 0.138 |

Full 0.05–0.95 sweep and confusion counts are in `external_results.json`. The 0.50 operating point was fixed before evaluation; no sweep point has been promoted to deployment.

## Access and task gaps

- i2b2/n2c2 clinical PHI and SecretBench require authorized agreements; access was unavailable. MIMIC-IV-Note also lacks PHI gold for this task. No clinical PHI or real-secret score is reported.
- Open de-identified glaucoma notes have no PHI gold, so a true clinical FPR/F1 cannot be inferred. Authoritative data-governance documents have no two-person human gold annotation here; governance remains unevaluated.
- TAB is English. JailbreakLLMs has no per-row language gold, so language-specific F1 cannot be computed reliably; the external headline makes no Turkish generalization claim. Prior Turkish project scores remain internal results.
- AgentDojo/BIPIA/InjecAgent are constructed agentic benchmarks and remain a separate tier, not mixed with this human-origin text headline.

## Reproduction

From this repository with its base `.venv` installed, a single command clones the pinned public sources, installs baseline-only dependencies, checks model/data versions and regenerates all results:

```bash
bash scripts/run_external_all.sh
```

Individual stages, if needed:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_external_real.py
PYTHONPATH=src .venv/bin/python scripts/run_external_baselines.py
PYTHONPATH=src .venv/bin/python scripts/run_neuraltrust_baseline.py
PYTHONPATH=src .venv/bin/python scripts/audit_jailbreak_baseline.py
PYTHONPATH=src .venv/bin/python scripts/calibrate_external_tab.py
PYTHONPATH=src .venv/bin/python scripts/run_external_ablation.py
PYTHONPATH=src .venv/bin/python scripts/run_clinical_unlabeled.py
PYTHONPATH=src .venv/bin/python scripts/write_external_report.py
```

Prediction archives under `benchmarks/external/predictions/` contain hashes, IDs, scores and evidence offsets, never raw text, and are intentionally gitignored. The source corpora are also gitignored. Official sources, versions, local dependencies and caveats are recorded in the dataset audit.

## Frozen component diagnostics

| Dataset | Component | F1 | FPR | FNR |
|---|---|---:|---:|---:|
| TAB | model_raw | 0.107 | 0.083 | 0.904 |
| TAB | model_existing_calibration | 0.107 | 0.083 | 0.904 |
| TAB | model_calibration_evidence | 0.107 | 0.083 | 0.904 |
| TAB | policy_non_allow_operational_proxy | 0.145 | 0.970 | 0.279 |
| JailbreakLLMs | model_raw | 0.202 | 0.900 | 0.055 |
| JailbreakLLMs | model_existing_calibration | 0.202 | 0.900 | 0.055 |
| JailbreakLLMs | model_calibration_evidence | 0.202 | 0.900 | 0.055 |
| JailbreakLLMs | policy_non_allow_operational_proxy | 0.197 | 0.986 | 0.003 |

TAB: 4 abstentions; policy actions {'ALLOW': 117, 'REDACT': 0, 'REVIEW': 105, 'BLOCK': 1857}.
JailbreakLLMs: 9 abstentions; policy actions {'ALLOW': 75, 'REDACT': 3, 'REVIEW': 559, 'BLOCK': 5251}.

`policy_non_allow_operational_proxy` counts REDACT, REVIEW and BLOCK as interventions across all risk tasks; its F1 is **not** a PII or jailbreak classifier score. Evidence boosts only the existing PII patterns. No component was fitted on the test set.

## De-identified clinical note diagnostic (unlabeled)

The public glaucoma repository contains 480 de-identified notes. Across 7179 nonoverlapping model windows, raw PII score ≥0.50 occurred in 4678 windows and 478 documents. **These are flag counts, not false-positive rates.** The corpus has medication annotations but no PHI gold; surrogate or residual identifiers may remain. No clinical PHI precision, recall or F1 is inferred.
