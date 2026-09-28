# TrustLaya-S v3 external scorecard — experimental, NO-GO

All v3 decision thresholds and Platt calibration parameters were selected on disjoint portions of **development data only**. V2 remains the shipped baseline. Values below use the same clean, near-duplicate-screened test rows for paired v2/v3 comparison; [frozen full-cohort v2 scores](v2_external_baseline_frozen.md) remain unchanged. Probabilities are task-model scores, not legal/ethical verdicts.

## PII: TAB official ECHR test, English, `558e09e`, DIRECT PERSON/CODE only

Clean test: **1,794 nonoverlapping 94-token windows; 103 positive**. V2 threshold 0.50 raw; v3 threshold 0.47 calibrated (selected on TAB dev). V3 token threshold 0.50 selected separately on TAB dev. Other TAB identifier classes are outside this binary projection.

| System and granularity | Precision | Recall | F1 | FPR | FNR | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 frozen, window presence | 0.115 | 0.165 | 0.135 | 0.077 | 0.835 | 0.629 | 0.088 |
| V3 token-head-derived, window presence | 0.306 | 0.107 | 0.158 | 0.015 | 0.893 | 0.860 | 0.274 |
| Presidio external baseline, window presence at 0.50 | 0.217 | 0.981 | 0.355 | 0.216 | 0.019 | 0.883 | 0.214 |

The v3 head ranks windows much better and greatly reduces false alarms, but misses 92/103 positive windows. It does **not** meet the PII recall criterion. Presidio is a valid comparator here only at **window presence**, not exact span: its binary presence score comes from any configured recognizer and can flag names that TAB marks QUASI or NO_MASK.

| V3 output granularity | PERSON F1 | CODE F1 | Overall F1 | Overall precision | Overall recall |
|---|---:|---:|---:|---:|---:|
| Token-level entity presence | 0.120 | 0.429 | 0.204 | 0.297 | 0.155 |
| Exact character span + type | 0.039 | 0.240 | 0.079 | 0.081 | 0.076 |
| Case/document binary presence (125 cases, 36 positive) | — | — | 0.581 | 0.692 | 0.500 |

The exact-span metric counts identical `(start,end,type)` tuples. It cannot be compared with Presidio's window-presence F1. The tokenizer/postprocessor is much better on legal application codes than person names. `DIRECT PERSON/CODE` is narrow relative to general PII; absent labels for EMAIL/PHONE/ADDRESS do not imply those entity types are safe.

## Direct jailbreak: JailbreakLLMs community test, `2dbd7bbc`, English/unspecified

Clean test: **5,761 prompts; 635 positive** after source and near-duplicate exclusions. Gandalf training labels are human-submitted **prompt-injection game attempts**, a related but distinct task. V2 threshold 0.50 raw; v3 threshold 0.30 calibrated, selected on Gandalf/prompts.chat dev.

| System | Precision | Recall | F1 | FPR | FNR | ROC-AUC | PR-AUC | False allow* | False block* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 frozen | 0.115 | 0.946 | 0.206 | 0.898 | 0.054 | 0.710 | 0.253 | 34 | 4,603 |
| V3 Gandalf + human benign head | 0.076 | 0.035 | 0.048 | 0.052 | 0.965 | 0.550 | 0.114 | 613 | 267 |
| NeuralTrust external baseline at 0.50 | 0.237 | 0.682 | 0.352 | 0.272 | 0.318 | 0.763 | 0.243 | 202 | 1,394 |

\*False allow/block are **binary-head proxies** if this single head alone controlled a block decision. Actual gateway policy uses other signals and was not rebenchmarked here. NeuralTrust training overlap is unknown and it is a comparator, not a v3 component. The v3 head fails transfer badly and **must not replace the deployed prompt-injection head**. By source, v3 recall is 0.032 Discord, 0.015 Reddit, 0.065 website; source differences are large.

## Prompt injection / agent security / Turkish / hard negatives

- The Gandalf development task is direct prompt-injection against a password game. It is **not** an independent direct-jailbreak result. Its high dev F1 (0.948 on the threshold-selection subset) did not transfer to the community jailbreak source.
- On Gandalf's official **112 attack-only test prompts** (no near overlaps with v3 train/dev at cosine >= 0.85), the v3 head flags **111/112 = 0.991** at the dev-selected threshold. This is a within-source detection fraction, **not F1, precision, or independent-source recall** because it has no benign control. The contrast with 22/635 JailbreakLLMs attacks illustrates the transfer failure.
- Indirect injection and agent-tool attacks were not newly measured in this v3 milestone. Earlier synthetic/agent benchmarks remain internal regression evidence, not new external scores.
- TAB, Gandalf and the retained JailbreakLLMs sample do not provide a reliable Turkish-language label partition. There is **no v3 Turkish external F1**. The v2 Turkish PII diagnostic used data involved in v2 development and is not substituted here.
- The 1,279 human/community benign training prompts include 137 broad security-vocabulary examples, but only 11 explicitly mention `jailbreak`, `prompt injection`, or `system prompt`; dev has 42 and 2 respectively. This hard-negative coverage is too thin for a broad claim.

See [calibration](v3_calibration.md), [error analysis](v3_error_analysis.md), [leakage audit](v3_data_leakage.md), [ablation](v3_ablation.md), and [machine-readable results](v3_external_results.json).
