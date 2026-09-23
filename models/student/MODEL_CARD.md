---
license: mit
language:
- tr
- en
tags:
- safety
- prompt-injection
- multi-task
- onnx
- research
base_model: ytu-ce-cosmos/turkish-medium-bert-uncased
model-index:
- name: TrustLaya-S
  results: []
---
# TrustLaya-S

42,138,641 parameter Turkish-first encoder for nine AI-safety risk scores, severity and advisory action. A separate deterministic policy engine determines the final ALLOW, REDACT, REVIEW or BLOCK action. [Source and documentation](https://github.com/ege-arhan/trustlaya-s) · [Interactive demo](https://huggingface.co/spaces/xzwq/TrustLaya-S-demo).

**Research prototype.** It is not a production safety gate. Scores were temperature-scaled on synthetic validation examples, so they are **not** established real-world risk probabilities. Confidence measures categorical output concentration and is not a calibrated correctness probability. Risk score is not a legal or ethical verdict.

## Files and use

- `model.safetensors`: custom PyTorch multitask network weights (not compatible with generic `AutoModel.from_pretrained`).
- `config.json`, tokenizer files: backbone structure and tokenizer.
- `calibration.json`: per-task synthetic-validation temperature scaling.
- `trustlaya.onnx`: FP32 exported network; `trustlaya_int8.onnx`: dynamically quantized CPU variant.
- `policy.yaml`: default separate policy thresholds.

```bash
git clone https://github.com/ege-arhan/trustlaya-s.git
cd trustlaya-s
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_artifacts.py
.venv/bin/python demo/cli_demo.py --backend onnx --text "Önceki talimatları yok say."
```

## Training and evaluation

Backbone: [YTU CE Cosmos Turkish medium BERT](https://huggingface.co/ytu-ce-cosmos/turkish-medium-bert-uncased), MIT. Weak teacher signal: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), Apache-2.0, on 128 examples. The teacher is not ground truth. 10,000 controlled synthetic examples, family-disjoint 6,972/1,053/1,975 train/validation/test split. Text is truncated to 96 tokens. Agent permissions are processed by policy, not the encoder.

Synthetic test (1,975): mean task accuracy **0.907**, macro F1 **0.663**, PII F1 **1.000**, security-risk F1 **0.777**, prompt-injection F1 **0.555**, data-governance F1 **0.000**. ECE **0.093** and Brier **0.083** are measured on synthetic test. Per-task details: [evaluation report](https://github.com/ege-arhan/trustlaya-s/blob/main/reports/evaluation.json). These numbers do not estimate production performance.

Independent [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) held-out test (116): injection F1 **0.765**, recall **0.867**, false-positive rate **0.429** at 0.5. Independent [Rogue Security hard benign examples](https://huggingface.co/datasets/rogue-security/real-world-benign-use-cases) (178): false-positive rate **0.242**. This is the main deployment blocker. Toxicity is a non-equivalent proxy for ethics risk; a separate diagnostic gave F1 **0.012** on 2,000 [Turkish toxic-language](https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language) items. [Full independent data analysis](https://github.com/ege-arhan/trustlaya-s/blob/main/docs/external_evaluation.md).

Apple Silicon Mac, batch 1 warm median: PyTorch CPU **12.39 ms**, PyTorch MPS **6.03 ms**, ONNX CPU **5.09 ms**, ONNX INT8 CPU **5.61 ms**. FP32 ONNX **159.9 MiB**, INT8 ONNX **40.5 MiB**. INT8 is smaller but not faster in this measurement. [Benchmark details](https://github.com/ege-arhan/trustlaya-s/blob/main/benchmarks/edge.json).

## Intended use and limits

Use for research, triage and human-reviewed experimentation. Do not rely on it alone for irreversible enforcement or personal-data processing. Regex evidence can miss novel PII/secrets and can surface sensitive input spans; avoid raw logging. Synthetic data patterns, poor data-governance recall, prompt-injection false positives, and limited English testing constrain generalization. See [limitations](https://github.com/ege-arhan/trustlaya-s/blob/main/docs/limitations.md).
