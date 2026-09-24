# TrustLaya-S final report (v1 historical run and v2 experimental update)

## Problem and goal

Compact edge-oriented AI safety decision support for Turkish-first text and AI agent permission metadata. This is a working MVP, not a production safety gate.

## Architecture and student

Pretrained Turkish BERT shared encoder, mean pooling, nine independent binary risk heads, severity head, and advisory action head. Total parameters: **42,138,641**; trainable during this run: **19,185,681**. Embeddings and first two encoder layers were frozen. 96-token input. Source backbone: [YTU CE Cosmos Turkish Medium BERT](https://huggingface.co/ytu-ce-cosmos/turkish-medium-bert-uncased), MIT license. Model output is calibrated per task, combined with deterministic evidence, then sent to an independent policy engine. The policy emits ALLOW, REDACT, REVIEW or BLOCK.

## Teacher and distillation

[convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), Apache-2.0 license; 321,908,998 local parameters. Local model and tokenizer SHA-256 match the public multilingual checkpoint. 128 training rows were queried for nine typed `noul` probabilities. There were 73 unique text keys in the teacher file. Student loss is binary cross-entropy for risk heads plus 0.3 severity cross-entropy plus 0.3 action cross-entropy; the weak teacher binary cross-entropy term has weight 0.05 only when teacher's binary side agrees with synthetic label. Teacher predictions are not ground truth. Training: 150 supervised steps and 80 resumed weak-distillation steps, batch 32. Logs: `logs/final_training.log` and `logs/final_distillation.log`.

## Dataset

10,000 synthetic samples, eleven categories, Turkish/English/mixed templates. Train 6972, validation 1053, test 1975. Template families and exact texts do not cross splits. Synthetic examples are controlled, but many examples share a small set of templates; results are not evidence of open-world robustness.

Published dataset: [ege-arhan/TrustLaya-S-synthetic](https://huggingface.co/datasets/ege-arhan/TrustLaya-S-synthetic). A fresh `datasets.load_dataset` call verified all three split counts and schema after upload. Independent evaluation corpora are linked and cited but not redistributed.

## Full held-out synthetic test (1975 examples)

Mean task accuracy: **0.907**; macro F1: **0.663**. Threshold is 0.5 after calibration and deterministic PII/secret score floors.

| Task | F1 | Recall | False positive rate | False negative rate | Positive support |
|---|---:|---:|---:|---:|---:|
| pii | 1.000 | 1.000 | 0.000 | 0.000 | 356 |
| secret | 1.000 | 1.000 | 0.000 | 0.000 | 201 |
| prompt_injection | 0.555 | 0.652 | 0.115 | 0.348 | 279 |
| dangerous_instruction | 0.662 | 0.495 | 0.000 | 0.505 | 196 |
| privacy_risk | 0.532 | 0.458 | 0.051 | 0.542 | 321 |
| security_risk | 0.777 | 0.915 | 0.594 | 0.085 | 1135 |
| ethics_risk | 0.443 | 0.284 | 0.000 | 0.716 | 341 |
| oversight_risk | 1.000 | 1.000 | 0.000 | 0.000 | 369 |
| data_governance_risk | 0.000 | 0.000 | 0.000 | 1.000 | 171 |

## Baselines on same 128-row test subset

The following scores share the same 128 examples. Teacher probabilities were not domain-calibrated. Rule-only baseline covers PII and secrets; other tasks predict zero. F1 and recall are macro means over nine tasks. Latency is batch 1, measured in the baseline script, with student on MPS and teacher on CPU.

| Model | Parameters | Macro F1 | Macro recall | Mean ECE | p50 latency ms | Weight size |
|---|---:|---:|---:|---:|---:|---:|
| teacher | 322M | 0.124 | 0.185 | 0.201 | 118.427 | 614.0 MiB |
| student | 42.1M | 0.618 | 0.612 | 0.108 | 12.711 | 160.8 MiB |
| rules | 0 | 0.222 | 0.222 | 0.146 | not measured | 0 |

The full 1,975-row test and the 128-row baseline subset must not be compared as if they were the same experiment. Rule-based PII/secret baseline F1 on the full test is 1.000/1.000, which reflects exact synthetic patterns.

## Calibration and uncertainty

Nine independent temperature scalars fitted on validation. Mean ECE raw **0.111**, calibrated **0.093**. Mean Brier raw **0.103**, calibrated **0.083**. Mean NLL raw **0.423**, calibrated **0.265**. Per-task raw and calibrated results are in `reports/calibration.json`. Calibration improved mean metrics but worsened ECE for some individual tasks. Confidence is the minimum of severity and action softmax peaks, a decisiveness heuristic rather than calibrated correctness. When confidence is below policy threshold 0.60, policy returns REVIEW and `abstain=true` unless deterministic evidence takes precedence. A 0.90 risk at 0.95 confidence can trigger BLOCK; a 0.90 risk at 0.55 confidence triggers REVIEW.

## Evidence and policy

Regex/pattern spans cover Turkish national ID checksum, phone, email, validated IP, Luhn card, IBAN, labeled address/name/customer ID, API keys, bearer tokens, passwords, database credentials and related secrets. PII external transfer triggers REDACT; secret evidence triggers BLOCK. Agent shell or credential access without human approval triggers REVIEW. Thresholds live in `configs/policy.yaml`. Model `model_action` remains advisory; final `action` is policy output.

## Export, size and latency

| Artifact | Size |
|---|---:|
| Student FP32 safetensors | 160.8 MiB |
| Student FP16 safetensors | 80.4 MiB |
| ONNX FP32 | 159.9 MiB |
| ONNX INT8 per-channel | 40.5 MiB |

| Backend | Cold start ms | Warm p50 ms, batch 1 | Approx RSS delta MiB |
|---|---:|---:|---:|
| pytorch_cpu | 520.047 | 12.388 | 173.062 |
| onnx_cpu | 137.119 | 5.087 | 287.625 |
| onnx_int8_cpu | 90.784 | 5.613 | 55.078 |
| pytorch_mps | 535.586 | 6.031 | 62.281 |

RSS deltas were measured sequentially in one process; allocator reuse and already loaded models make them approximate and unsuitable for direct backend comparison. INT8 changed **5.078%** of final actions versus FP32 on 256 synthetic test rows. INT8 is experimental and should not be deployed without a stricter parity gate. Detailed F1 differences are in `reports/quantization.json`.

## Test and verification

The original v1 run passed 11 pytest tests. ONNX checker and ONNX CPU inference passed. CLI produced REDACT for the requested Turkish PII transfer example. Stage outcomes are in `logs/final_check.log`. No Arduino UNO Q board was connected or measured.

Published [GitHub source](https://github.com/ege-arhan/trustlaya-s), [Hugging Face v1 model](https://huggingface.co/ege-arhan/TrustLaya-S), and [Space](https://huggingface.co/spaces/ege-arhan/TrustLaya-S-demo). The v1 Space was manually checked with PII transfer (REDACT), prompt injection (BLOCK), and benign text (ALLOW).

## Limitations and next work

Risk score is not a legal or ethical verdict. Probabilities are task-model outputs and require task-specific calibration. Data governance F1 is zero on the held-out synthetic templates. Dangerous-instruction and ethics false negative rates are high. Teacher is weak on the specialized schema. Template-generated samples overstate pattern performance and do not cover natural incidents, adversarial paraphrases, or shifted populations. Confidence is not calibrated correctness. English support is limited by a Turkish-first backbone. Next priority: independently labeled real Turkish/English validation and test sets, hard negative collection, then threshold and calibration refit. Only after that, revisit INT8 and UNO Q deployment.

## Independent evaluation update (2026-09-24)

See `docs/external_evaluation.md` and `reports/external_evaluation.json`. Held-out deepset prompt injection F1 is 0.765 with false-positive rate 0.429. A hard benign set from real coding-agent traffic had false-positive rate 0.242. Turkish toxicity proxy performance is near zero; toxicity is not the same as this model's ethics-risk task. A reviewed incident set is positive-only, so its 1.000 any-risk coverage cannot establish discrimination. Extra training and real-data head fine-tuning were evaluated on development splits and rejected because they did not improve the combined quality gate. The baseline student remains the published checkpoint.

## Further independent diagnostics and policy change

An independent Turkish privacy dataset gave PII model F1 0.803 and regex-plus-model F1 0.880 on 2,000 balanced, supported-category examples. An independent multilingual secret benchmark gave model F1 0.676 with a 0.949 false-positive rate on 2,000 balanced synthetic/augmented examples. The rule detector alone had 0.073 false-positive rate but 0.403 recall. Because model-only secret scores overfire on benign technical strings, the policy now sends those hits to REVIEW; explicit secret pattern evidence still BLOCKs. This policy change does not change model weights or the recorded synthetic test metrics. Sources, sampling and per-type results: `docs/external_evaluation.md` and `reports/external_privacy_secret.json`.

A further independent development/test secret threshold check gave test ROC AUC 0.536; the development-selected threshold near 1.0 still yielded FPR 0.837. Threshold adjustment did not justify restoring model-only auto-blocking. See `reports/secret_threshold_diagnostic.json`.

## Advanced v2 candidate update

The versioned advanced candidate preserves v1 and updates only the PII head using frozen encoder features from disjoint Turkish privacy scenarios. A distinct 2,000-row synthetic Turkish PII test gave v2 F1 0.784 and false-positive rate 0.316. The original 1,975-row synthetic mixed test remains macro F1 0.663; injection F1 0.555 and data-governance F1 zero. Current confidence fails selective-risk testing. A 24-family adversarial suite found F1 zero for encoded attacks. FP32 ONNX remains 159.9 MiB; INT8 changed 5.1% of policy actions on 256 rows. A local session-aware API and agent risk policy are implemented. The complete scoped evidence, methods, and limitations are in `reports/research_report.md`, `MODEL_CARD.md`, and `DATA_CARD.md`. `scripts/final_check.sh` passed all stages, including 17 pytest tests, in the advanced branch. No production readiness or UNO Q hardware validation is claimed.

## Latest independent diagnostic and release verification

On the same 2,000-row Turkish PII mapping, rules alone reached F1 0.480 and FPR 0; v1/v2 hybrid F1 was 0.737/0.784, but v2 FPR was 0.316. The source is synthetic. A separate 182-case, hand-crafted agentic prompt-injection benchmark gave v2 tool-result classification F1 0.484, recall 0.380, FPR 0.675 and false-negative rate 0.620. Overlapping 94-token windows improved F1 to 0.592, with FPR still 0.675. This measures text detection only, not attack success. These failures rule out autonomous enforcement. The policy now requires REVIEW for declared lower-trust tool output entering an unapproved privileged agent; callers must provide truthful metadata and pause execution. Full scopes, pinned source revisions and breakdowns are in `reports/independent_pii_v2.json` and `reports/agent_injection_independent.json`.

The latest saved Mac batch-one warm p50 was 13.864 ms PyTorch CPU, 7.311 ms PyTorch MPS and 7.388 ms ONNX CPU FP32; measurements vary between runs. The pinned PII training data reproduced the published v2 safetensors SHA-256 exactly in a separate candidate directory. The [experimental GitHub prerelease](https://github.com/ege-arhan/trustlaya-s/releases/tag/v2.0.0-rc1) was downloaded with SHA-256 verification and run from a clean checkout. The [advanced Hugging Face repository](https://huggingface.co/ege-arhan/TrustLaya-S-Advanced) now hosts v2 safetensors, FP32/INT8 ONNX, tokenizer, calibration and policy. The three large Hugging Face file hashes match `models/advanced_manifest.json`; nine files were redownloaded and checksum-verified, and INT8 CLI inference returned `REDACT` for the Turkish PII-transfer example. See `reports/release_verification.md`. Latest `scripts/final_check.sh` passed all stages, including independent diagnostics, ONNX parity, 18 tests and CLI inference.

## Injection-head improvement experiments

An isolated BPI-only head reached F1 0.870 on its 2,961-row transformed test versus released v2 0.561. Joint training with legacy synthetic anchors reached 0.869. On a newly reserved 550-row Turkish/English PolyGuardBench check, the joint head reached F1 0.868 versus 0.486, with attack recall 0.989 versus 0.563 and benign FPR 0.153 versus 0.397. However, the joint head reduced the original mixed-only synthetic injection F1 from 0.555 to 0.464 and raised benign AgentInjectionBench tool-output FPR from 0.675 to 0.800. High F1 on that attack-heavy agent test hides the false alarms. The improvement is therefore domain-specific; no new injection checkpoint was promoted or uploaded. Exact test scopes, pinned source revisions and limitations are in `reports/injection_experiments.md`.

Adding a separate paired agentic training source improved its untouched 545-row synthetic test: F1 **0.628→0.872**, attack recall **0.832→0.929**, and benign FPR **0.864→0.211**. The candidate missed 20 attacks and flagged 56 benign cases there. On the independent broad PromptWall diagnostic, recall fell **0.723→0.698** while FPR fell **0.231→0.062**. Previously inspected AgentInjectionBench benign FPR stayed **0.800**. This candidate also remains unpromoted. The released v2 model and GitHub prerelease weights are unchanged.

A new pinned Bordair live-game source contains real human-written red-team submissions. On its 855 recorded bypass strings, v2 flagged 71.3% and the agentic candidate 36.4% at their own thresholds. Since the source is attack-intent-only and includes conversational strings, these fractions are not recall or F1. This diagnostic strengthens the decision not to promote the synthetic-data winner; see `reports/live_redteam_diagnostic.json`.

On a separate untouched 942-row NeurAlchemy grouped test with broad injection/jailbreak labels and benign examples, agentic candidate F1 was 0.805 versus v2 0.632, recall 0.726 versus 0.498, and both benign FPR 0.110. No normalized exact text overlapped our training/development sets, but semantic-family overlap remains unverified. This stronger broad-label result does not cancel the negative human-game diagnostic or 0.800 AgentInjectionBench benign FPR. See `reports/neuralchemy_cross_source.json`.

The [agentic INT8-only experimental prerelease](https://github.com/ege-arhan/trustlaya-s/releases/tag/v2.1.0-agentic-rc1) contains nine checksum-pinned assets. A fresh checkout downloaded them and completed ONNX INT8 CLI inference. FP32 **agentic** candidate weights and ONNX are local reproducible artifacts, not in that prerelease because large-file GitHub uploads stalled. The Hugging Face Advanced repository contains the separate **v2** model, not this unpromoted agentic head. See `reports/release_verification.md`.
