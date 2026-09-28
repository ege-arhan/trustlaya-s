# Adversarial evaluation

`scripts/evaluate_adversarial.py` evaluates 24 researcher-authored base cases, each with nine transformations: paraphrase, typo, Unicode, spacing, encoding, translation, mixed language, indirect wording, and role manipulation. The 240-row benchmark is controlled synthetic data, not a human-verified attack corpus. Related variants share a family and must never be split across train and test.

Measured prompt-injection F1 is 0.700 on 24 originals and 0.750 on 216 transformed cases. This aggregate is misleading: encoding has F1 **0.000** and false-negative rate **1.000** on 12 positive cases. Indirect wording has false-positive rate 0.667 on 12 negatives; mixed language has 0.417. Full counts and rates are in `reports/adversarial.json`.

Reproduce with `python scripts/evaluate_adversarial.py`. New external corpora must be checked for license, duplicates, task definition, and contamination before use. Decoded or tool-returned text should be inspected by a separate trust-boundary policy; the current text classifier does not reliably recognize encoded attacks.

## Independent agentic tool-output diagnostic

An Apache-2.0 [AgentInjectionBench](https://huggingface.co/datasets/ppradyoth/AgentInjectionBench) release supplied 142 attack cases and 40 matched benign controls. `scripts/evaluate_agent_injection.py` scores concatenated `tool_result` text with the v2 FP32 ONNX classifier. At 0.5, F1 was **0.484**, recall **0.380**, false-negative rate **0.620**, and false-positive rate **0.675**; balanced accuracy was **0.353**. Appending advertised tool descriptions did not materially help (F1 **0.486**, FPR **0.650**). The 96-token input limit truncated 129 of 182 tool-result texts.

A parameter-free diagnostic scanned overlapping 94-token windows and took the maximum score across them. This raised F1 to **0.592** and recall to **0.500** across 527 windows, but FPR stayed **0.675**. It is not promoted into inference or policy. The benchmark tests text detection, not whether an agent follows an attack; the source's tool context and execution contract are richer than this model's text input. The released cases have been inspected and are no longer untouched for future tuning. Exact breakdowns and source SHA-256 are in `reports/agent_injection_independent.json`.
