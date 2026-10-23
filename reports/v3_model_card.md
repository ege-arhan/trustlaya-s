# TrustLaya-S v3 experimental model card

**Release decision: NO-GO.** V3 is a local research candidate. V2 remains the released model and the unchanged gateway input. V3 must not be described as production-ready, secure, or as solving prompt injection.

## Architecture and files

The 42,138,641-parameter v2 Turkish BERT multitask model is loaded by SHA-256 `99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c`. A 2,565-parameter BIO token head and a 513-parameter attack head share its frozen 512-wide encoder. Total parameters when both heads are attached: **42,141,719**. The local `models/trustlaya-s-v3/` directory contains `pii_token_head.safetensors`, `attack_head.safetensors`, tokenizer, development calibration, thresholds and manifest. It **depends on the v2 base checkpoint**; it is not a standalone ONNX release. The v2 ONNX model does not contain the v3 heads.

## Training and intended use

TAB official train quality-checked ECHR cases trained `O/B/I-PERSON` and `O/B/I-CODE` tags for `DIRECT` identifiers. Gandalf human-submitted game prompts provided weak positive prompt-injection labels; prompts.chat CC0 community prompts provided conservative weak benign labels. Only head weights were trained. Seed 20260925, max length 96, 8 PII epochs (AdamW 0.002, batch 128 weighted CE), 15 attack epochs (AdamW 0.003, batch 128 balanced BCE). No final test text was included in fitting; exclusions and the later test inspection are disclosed in [leakage](v3_data_leakage.md).

The head is suitable for research on task granularity and cross-source failure. It is **not suitable for autonomous blocking or allowing of side-effecting agent operations**. The existing deterministic policy and authorization gate remain separate and unchanged.

## Evaluation and limitations

See [scorecard](v3_external_scorecard.md) for paired external results. TAB exact span F1 is only 0.079; the PERSON category is especially weak. The attack head identifies only 22/635 positive independent jailbreak prompts. Gandalf positive labels are similarity-filtered attack attempts, not independently adjudicated per prompt. prompts.chat benign labels are inferred from community prompt context plus a small exclusion filter, not manual gold. The classifier is English-dominated; no external Turkish v3 result exists. The final test was viewed during experimental iteration, so a fresh blind holdout is needed.

Risk score is not a legal or ethical verdict. Probabilities are task-model outputs and require task-specific calibration. Platt calibration fitted on the source-specific development data failed to transfer cleanly to JailbreakLLMs. The model should not replace v2 in the CLI, API, gateway or Arduino UNO Q deployment plan. No UNO Q hardware run was performed for v3.
