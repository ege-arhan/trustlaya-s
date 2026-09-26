---
language:
- tr
- en
license: mit
library_name: onnx
tags:
- safety
- prompt-injection
- multi-task
- research
- experimental
base_model: ytu-ce-cosmos/turkish-medium-bert-uncased
datasets:
- yusuf-said/turkish-privacy-filter-dataset
---
# TrustLaya-S Advanced: experimental v2 PII candidate

42,138,641 parameter Turkish-first multitask encoder. This version updates **only the PII head** of [TrustLaya-S v1](https://huggingface.co/ege-arhan/TrustLaya-S). It adds no new teacher distillation. A separate deterministic policy engine, agent/session risk rules, and evidence extraction live in the [source branch](https://github.com/ege-arhan/trustlaya-s/tree/feature/trustlaya-advanced). The model alone does not implement the full policy system.

**Research candidate only.** Not production-ready; do not use as sole gate for irreversible agent actions, personal-data disclosure, or legal/ethical decisions. Risk scores are task-model outputs, not validated real-world event probabilities. `confidence` is not calibrated correctness.

## Independent evaluation

A separate CC-BY 4.0 synthetic Turkish PII test sample (n=2,000; 1,000 task positives and 1,000 task-specific negatives) gave v2 hybrid PII F1 **0.784**, recall **0.848**, and false-positive rate **0.316** at a development-selected threshold of 0.8. This high false-positive rate blocks deployment. On the original synthetic mixed-only 1,975-row test, macro F1 **0.663**, security F1 **0.777**, injection F1 **0.555**, data-governance F1 **0.000**. Encoded prompt injection F1 **0.000** on a small controlled suite. The 33-case language smoke suite is too small to establish multilingual performance.

Original English-only synthetic validation mean raw/calibrated ECE **0.105/0.089**, Brier **0.103/0.085**, NLL **0.424/0.271**. Eight temperatures were evaluated on the same data used to fit them. Turkish PII test ECE **0.181**. More detail: [research report](https://github.com/ege-arhan/trustlaya-s/blob/feature/trustlaya-advanced/reports/research_report.md), [model card](https://github.com/ege-arhan/trustlaya-s/blob/feature/trustlaya-advanced/MODEL_CARD.md), [data card](https://github.com/ege-arhan/trustlaya-s/blob/feature/trustlaya-advanced/DATA_CARD.md).

## Firewall runtime demo (unchanged weights)

The same `trustlaya_s.onnx` now runs inside a Docker firewall demo on the [feat/v2-firewall-e2e branch](https://github.com/ege-arhan/trustlaya-s/tree/feat/v2-firewall-e2e): agent → gateway (this model + policy + single-use authorization) → trusted adapter → SQLite `record.write` target, on internal networks, with no torch in the runtime image. Analysis responses now report reading coverage: the model reads only the first 94 content tokens (`head_94_v1`), and an unread remainder turns ALLOW/REDACT into REVIEW for protected actions. Targets are idempotent per `operation_id`; a lost target reply is reported as `unknown`, not "not executed". See the [demo documentation](https://github.com/ege-arhan/trustlaya-s/blob/feat/v2-firewall-e2e/docs/firewall_demo.md).

All 13 fixed scenarios passed on 2026-09-26 (Docker on Apple Silicon): no unauthorized scenario created a target record, a retried operation never created a second record, and no key or secret appeared in logs. End-to-end p95 53 ms, model p95 45 ms over 100 sequential writes; gateway memory about 270 MiB. These are fixed demonstration inputs, not a quality benchmark. Observed model limits in the same runs: many short benign notes received REVIEW, and a 113-token benign meeting note was BLOCKed as prompt injection. No UNO Q measurement.

## Files and usage

`model.safetensors` is a custom nine-risk-head PyTorch model, not a generic `AutoModel`. `trustlaya_s.onnx` is FP32; `trustlaya_s_int8.onnx` is **experimental** and changed 5.1% of final policy actions on 256 synthetic rows. `calibration.json`, `decision_thresholds.json`, `policy.yaml`, and tokenizer files support the full pipeline. Batch-1 MacBook ONNX CPU p50 **4.629 ms** in the latest run; no Arduino UNO Q hardware benchmark was performed.

Use the [source code](https://github.com/ege-arhan/trustlaya-s/tree/feature/trustlaya-advanced) and its `Analyzer` with this model directory and an explicit ONNX path. Provenance: MIT [YTU Turkish BERT backbone](https://huggingface.co/ytu-ce-cosmos/turkish-medium-bert-uncased); weak v1 teacher was Apache-2.0 [Laya Multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), never ground truth. V2 PII training source is MIT [Turkish Privacy Filter Dataset](https://huggingface.co/datasets/yusuf-said/turkish-privacy-filter-dataset). No source examples are redistributed here.
