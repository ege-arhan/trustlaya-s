# TrustLaya-S Advanced research report

## Problem and objective

Build a compact, auditable text and agent-safety decision system for Turkish-first edge use. The system should expose model scores, evidence, uncertainty, and a deterministic policy action. It does not claim to make legal or ethical judgments.

## Prior art and sources

The original [Laya](https://huggingface.co/convaiinnovations/laya) and the specific [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) checkpoint differ; the latter is the weak teacher in v1. [Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M) shows a separate compact prompt-injection detector approach. [Open Policy Agent](https://www.openpolicyagent.org/docs) and [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview) illustrate separating declarative policy or guardrails from model output. This implementation uses a local Python policy; it does not claim feature parity with either framework.

## Current state and architecture

The v1 snapshot is in `docs/current_state.md`. V1 is a 42,138,641-parameter Turkish BERT with nine binary risk outputs plus severity/action heads. V2 preserves the backbone and all heads except PII. A frozen-encoder logistic regression PII head was trained on a separate MIT-licensed Turkish privacy corpus. An evidence engine checks explicit patterns. Deterministic agent permission and bounded session-chain logic contribute policy triggers; rule-constrained fusion combines risk signals without pretending the result is a probability. The final action comes from policy, not the model's action head. See `docs/architecture.md`.

## Data and leakage controls

The 10,000-row v1 synthetic set contains only 2,682 exact unique texts and 143 normalized prototypes. Family and exact text overlap across splits are zero, but train is Turkish/English, validation English-only, and test mixed-only. The v2 privacy source uses scenario-disjoint splits, 3,653 train, 478 development, 979 test. The separate 2,000-row CC-BY Turkish PII test was unused for training or threshold fitting; it was inspected after scoring. Adversarial and language smoke suites were researcher-authored. Source revisions, licenses and restrictions are in `DATA_CARD.md`.

## Teacher and distillation

The v1 local teacher is `convaiinnovations/laya-multilingual`, 321,908,998 parameters; model and tokenizer hashes match the public checkpoint. V1 used soft outputs only as a weak auxiliary term when compatible with synthetic labels. V2 PII training did **not** run teacher distillation; it trained a supervised head on frozen v1 features. Teacher predictions are never ground truth.

## Selection and measured results

Candidate PII regularization and threshold were selected on development scenarios with a false-positive constraint. Its 979-row scenario test PII F1 rose from 0.701 (v1 with development-selected threshold) to 0.923, while false-positive rate rose from 0.0067 to 0.0403. The distinct 2,000-row Turkish PII test is less favorable: v1 F1 0.737, v2 F1 0.784, and v2 FPR 0.316. Dates account for 312/500 v2 false positives; private person and account number detections are 62/125 and 68/125. Future tuning needs a new untouched holdout.

On 1,975 synthetic test rows: mean task accuracy 0.9066; macro F1 0.6632; security F1 0.7769; PII F1 1.0000; injection F1 0.5549; data-governance F1 0.0000. High synthetic PII accuracy conflicts with the independent PII result and should not be headlined alone. The 33-case TR/EN/mixed smoke suite has only 11 examples per language slice, so it cannot establish broad language support.

## Calibration and abstention

On the English-only synthetic validation set, mean ECE improved from 0.1051 to 0.0890, Brier from 0.1027 to 0.0853, NLL from 0.4243 to 0.2714. Eight temperatures were evaluated on the data used to fit them. Separate Turkish PII test ECE is 0.181. Action `confidence` median is 0.9988 on synthetic test, but action error among retained examples increases when confidence threshold rises. The present confidence output is not a reliable correctness estimate. Policy REVIEW/BLOCK rules remain the safety fallback.

## Adversarial, quantization and edge

The 24-family adversarial suite yields clean injection F1 0.700 and transformed F1 0.750, but base64 encoding F1 0.000. FP32 ONNX max risk-logit difference from PyTorch on 256 rows is 5.05e-05. INT8 max risk-logit difference is 2.429, and 5.1% of final policy actions differ from FP32. Experimental INT8 is not an approved replacement. FP32 ONNX size is 159.9 MiB; MacBook batch-1 ONNX CPU median is 4.629 ms. CPU, MPS, and ONNX details are in `benchmarks/results.json`. No Arduino UNO Q hardware was tested; the Linux application processor is the prospective target.

## API, reproducibility and limitations

The local HTTP API accepts text, agent permissions, and optional session ID; it stores bounded session history without raw prompt text and can emit an evidence-redacted audit log. It is not network-secure by itself. `scripts/final_check.sh` runs the non-destructive validation gate; `docs/training.md` gives rebuild commands. Major blockers: independent PII FPR, encoded injection misses, secret false positives, nonfunctional data-governance head, uncalibrated confidence, synthetic/shifted evaluation, and INT8 action drift. See `docs/limitations.md`.

## Next experiments

Acquire licensed, human-reviewed, genuinely independent TR/EN/mixed positives and hard negatives. Retrain injection, secret, and data-governance heads with group-held-out evaluation. Fit correctness confidence on untouched development data and enforce a measured selective-risk curve. Validate policy thresholds for organization-specific costs. Benchmark ONNX FP32 and any quantized candidate on the actual UNO Q Linux image before deployment.
