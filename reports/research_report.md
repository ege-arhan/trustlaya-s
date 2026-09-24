# TrustLaya-S Advanced research report

## Problem and objective

Build a compact, auditable text and agent-safety decision system for Turkish-first edge use. The system should expose model scores, evidence, uncertainty, and a deterministic policy action. It does not claim to make legal or ethical judgments.

## Prior art and sources

The original [Laya](https://huggingface.co/convaiinnovations/laya) and the specific [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) checkpoint differ; the latter is the weak teacher in v1. [Prompt Guard 86M](https://huggingface.co/meta-llama/Prompt-Guard-86M) shows a separate compact prompt-injection detector approach. [Presidio](https://microsoft.github.io/presidio/) supplies established PII recognition/anonymization patterns. [Open Policy Agent](https://www.openpolicyagent.org/docs) and [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/about-nemo-guardrails-library/overview) illustrate separating policy or guardrails from model output, including agent tool controls. TrustLaya-S combines task heads with local evidence, permission/session rules and a small edge-oriented runtime; it does not claim novel detectors, agent execution protection or feature parity with these systems.

## Current state and architecture

The v1 snapshot is in `docs/current_state.md`. V1 is a 42,138,641-parameter Turkish BERT with nine binary risk outputs plus severity/action heads. V2 preserves the backbone and all heads except PII. A frozen-encoder logistic regression PII head was trained on a separate MIT-licensed Turkish privacy corpus. An evidence engine checks explicit patterns. Deterministic agent permission and bounded session-chain logic contribute policy triggers; rule-constrained fusion combines risk signals without pretending the result is a probability. The final action comes from policy, not the model's action head. See `docs/architecture.md`.

## Data and leakage controls

The 10,000-row v1 synthetic set contains only 2,682 exact unique texts and 143 normalized prototypes. Family and exact text overlap across splits are zero, but train is Turkish/English, validation English-only, and test mixed-only. The v2 privacy source uses scenario-disjoint splits, 3,653 train, 478 development, 979 test. The separate 2,000-row CC-BY Turkish PII test was unused for training or threshold fitting; it was inspected after scoring. Adversarial and language smoke suites were researcher-authored. Source revisions, licenses and restrictions are in `DATA_CARD.md`.

## Teacher and distillation

The v1 local teacher is `convaiinnovations/laya-multilingual`, 321,908,998 parameters; model and tokenizer hashes match the public checkpoint. V1 used soft outputs only as a weak auxiliary term when compatible with synthetic labels. V2 PII training did **not** run teacher distillation; it trained a supervised head on frozen v1 features. Teacher predictions are never ground truth.

## Selection and measured results

Candidate PII regularization and threshold were selected on development scenarios with a false-positive constraint. Its 979-row scenario test PII F1 rose from 0.701 (v1 with development-selected threshold) to 0.923, while false-positive rate rose from 0.0067 to 0.0403. The distinct 2,000-row Turkish PII test is less favorable: v1 F1 0.737, v2 F1 0.784, and v2 FPR 0.316. Dates account for 312/500 v2 false positives; private person and account number detections are 62/125 and 68/125. Future tuning needs a new untouched holdout.

On 1,975 synthetic test rows: mean task accuracy 0.9066; macro F1 0.6632; security F1 0.7769; PII F1 1.0000; injection F1 0.5549; data-governance F1 0.0000. High synthetic PII accuracy conflicts with the independent PII result and should not be headlined alone. The 33-case TR/EN/mixed smoke suite has only 11 examples per language slice, so it cannot establish broad language support.

The same independent BTX24 sample also supplies a common-scope PII rule baseline: F1 0.480, recall 0.316 and FPR zero. V1/v2 hybrid F1 is 0.737/0.784 respectively, but their false-positive rates are 0.474/0.316. These are task-specific comparisons, not a global safety leaderboard.

An additional Apache-2.0 agentic injection benchmark has 142 attacks and 40 matched benign tool-output controls. The v2 prompt-injection head on concatenated tool results achieved F1 0.484, recall 0.380, FPR 0.675 and balanced accuracy 0.353 at 0.5. Of 182 texts, 129 exceed the 96-token input window. A 94-token overlapping-window maximum diagnostic improved F1 to 0.592 and recall to 0.500 while leaving FPR at 0.675; it was not promoted. This test cannot establish agent attack success or failure because it only scores text. See `reports/agent_injection_independent.json`.

An isolated BPI-trained injection-head experiment then improved a new Turkish/English PolyGuardBench cross-axis diagnostic to F1 0.868 from 0.486, with attack recall 0.989 versus 0.563 and benign FPR 0.153 versus 0.397. The same joint candidate regressed on the original mixed-only synthetic test (F1 0.464 versus 0.555) and produced 0.800 false-positive rate on benign AgentInjectionBench tool returns. Adding a separate paired agentic training source yielded F1 0.872 and FPR 0.211 on its scenario-isolated synthetic test versus v2 F1 0.628 and FPR 0.864. Yet it still flagged 80% of benign AgentInjectionBench tool returns and lost broad PromptWall attack recall (0.723 to 0.698). Neither candidate was promoted. BPI's labels cover broader adversarial prompts than strict injection, and PolyGuardBench's attack/benign rows come from different axes. Full counts and experiment selection are in `reports/injection_experiments.md`.

## Calibration and abstention

On the English-only synthetic validation set, mean ECE improved from 0.1051 to 0.0890, Brier from 0.1027 to 0.0853, NLL from 0.4243 to 0.2714. Eight temperatures were evaluated on the data used to fit them. Separate Turkish PII test ECE is 0.181. Action `confidence` median is 0.9988 on synthetic test, but action error among retained examples increases when confidence threshold rises. The present confidence output is not a reliable correctness estimate. Policy REVIEW/BLOCK rules remain the safety fallback.

## Adversarial, quantization and edge

The 24-family adversarial suite yields clean injection F1 0.700 and transformed F1 0.750, but base64 encoding F1 0.000. FP32 ONNX max risk-logit difference from PyTorch on 256 rows is 5.05e-05. INT8 max risk-logit difference is 2.429, and 5.1% of final policy actions differ from FP32. Experimental INT8 is not an approved replacement. FP32 ONNX size is 159.9 MiB; MacBook batch-1 ONNX CPU median is 7.388 ms in the latest saved run. CPU, MPS, and ONNX details are in `benchmarks/results.json`. No Arduino UNO Q hardware was tested; the Linux application processor is the prospective target.

## Ablations and rejected training

The base BERT checkpoint has no safety heads, so a base-only macro F1 is **NOT MEASURED**. The v1 multi-task model and v2 PII-head candidate share a 1,975-row synthetic test macro F1 of 0.663; v2 differs only in PII. A common 2,000-row Turkish PII test compares rules (F1 0.480), v1 hybrid (0.737) and v2 hybrid (0.784), with different recall/FPR tradeoffs detailed above. Raw versus temperature-scaled mean ECE is 0.105 versus 0.089 on the English-only synthetic validation set, mostly in-sample. Full factorial `base + multitask + adversarial training + calibration` variants were **NOT MEASURED**. An earlier real-data injection head fine-tune and blend were rejected on their development gate because apparent internal F1 gains coincided with worse false-positive behavior; see `reports/experiment_decisions.md`. No adversarially trained checkpoint is claimed here. Full system policy outcomes are not comparable to per-head F1 without a matching action-labeled benchmark.

## Evidence, agent and session policy

Pattern extraction returns type, exact text and offsets for detected PII, secrets and external transfer. This is not a learned NER head, and name/account-number coverage remains incomplete. Permission scoring reports shell, filesystem, network, database and credential subrisks. Bounded session state recognizes credential-access/external-transfer and filesystem/shell/external chains without storing raw prompts. These deterministic signals feed a rule-constrained fusion index and policy action; neither index is calibrated event probability. Because the independent agentic injection benchmark showed poor text detection, `untrusted_tool_output` from a privileged agent without human approval now requires REVIEW regardless of model score; callers must provide truthful metadata and enforce the resulting pause. An optional audit log records risk, action, triggered rule and evidence offsets but omits evidence text. The HTTP API rejects per-request policy overrides and binds to localhost by default.

## API, reproducibility and limitations

The local HTTP API accepts text, agent permissions, and optional session ID; it stores bounded session history without raw prompt text and can emit an evidence-redacted audit log. It is not network-secure by itself. `scripts/final_check.sh` runs the non-destructive validation gate; `docs/training.md` gives rebuild commands. Major blockers: independent PII FPR, encoded injection misses, secret false positives, nonfunctional data-governance head, uncalibrated confidence, synthetic/shifted evaluation, and INT8 action drift. See `docs/limitations.md`.

## Next experiments

Acquire licensed, human-reviewed, genuinely independent TR/EN/mixed positives and hard negatives. Retrain injection, secret, and data-governance heads with group-held-out evaluation. Fit correctness confidence on untouched development data and enforce a measured selective-risk curve. Validate policy thresholds for organization-specific costs. Benchmark ONNX FP32 and any quantized candidate on the actual UNO Q Linux image before deployment.
