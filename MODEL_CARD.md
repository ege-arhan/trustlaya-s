# TrustLaya-S Advanced v2 PII candidate

## Description and architecture

This is a **research candidate**, not the public v1 model. The 42,138,641-parameter Turkish BERT encoder and nine risk, severity, and advisory-action logits remain architecturally unchanged. The v2 checkpoint changes only the PII linear-head row, trained on a separate Turkish privacy corpus with synthetic anchors. The inference pipeline adds raw/calibrated/effective score separation, deterministic agent and session risk, three explicit fusion alternatives, evidence spans, abstention, and a separate policy engine. Agent and session scores are **policy heuristics, not learned or calibrated probabilities**.

Base: [YTU CE Cosmos Turkish medium BERT](https://huggingface.co/ytu-ce-cosmos/turkish-medium-bert-uncased), MIT. Weak v1 teacher: [Laya multilingual](https://huggingface.co/convaiinnovations/laya-multilingual), Apache-2.0; only 128 synthetic rows queried, 73 unique text keys, teacher output never ground truth. V2 PII training uses no new teacher targets. See [data card](DATA_CARD.md) and [training notes](docs/training.md).

## Intended and out-of-scope use

Intended for local safety triage research, controlled demonstrations, and human-reviewed policy experiments. Do **not** use as a sole control for blocking users, releasing private data, irreversible agent actions, legal decisions, or ethical verdicts. Do not call scores verified real-world event probabilities. Turkish is prioritized; English and mixed-language behavior is limited and uneven. Keep raw PII and credentials out of logs and the public Space.

## Measured results

| Test | V1 | V2 candidate | Scope |
|---|---:|---:|---|
| Macro F1 | 0.663 | 0.663 | Original 1,975-row synthetic, mixed-labeled test |
| PII F1 | 1.000 | 1.000 | Same synthetic test; strong template memorization risk |
| Hybrid PII F1 | 0.701 | 0.923 | 979 scenario-held-out synthetic/curated Turkish rows; thresholds selected on development scenarios |
| Hybrid PII FPR | 0.007 | 0.040 | Same scenario-held-out set |
| Hybrid PII F1 | 0.737 | 0.784 | Separate, synthetic [BTX24](https://huggingface.co/datasets/BTX24/turkish-privacy-pii-ner) test sample, n=2,000; v2 threshold 0.8 |
| Hybrid PII FPR | 0.474 | 0.316 | Same BTX24 sample; still too high |

On the original synthetic validation, v2 mean ECE was **0.089** calibrated versus **0.105** raw; mean Brier **0.085** versus **0.103**, NLL **0.271** versus **0.424**. Eight task temperatures were fitted on this same validation set. PII temperature was fitted on separate privacy development scenarios. These are in-sample diagnostics, not field calibration. [Reliability diagram](reports/reliability_v2.svg) and [per-task metrics](reports/calibration_v2.json).

Controlled prompt-injection stress test: clean F1 **0.700** (24), transformed F1 **0.750** (216), but base64-encoded variant F1 **0**. External v1 injection false-positive rate was 0.429 on deepset held-out data. Independent model-only secret false-positive rate was 0.949 on a different noncommercial corpus. The v2 PII update does not change injection or secret weights.

FP32 ONNX is 159.9 MiB and had maximum risk-logit drift 0.000051 versus PyTorch on 256 examples. INT8 ONNX is 40.5 MiB but changed **5.08%** of final policy actions on that sample. Mac batch-one warm p50: PyTorch CPU **11.685 ms**, MPS **4.584 ms**, ONNX CPU **4.629 ms**. Measurements are device and run specific. [Benchmark table](benchmarks/results.md).

## Inference example

```bash
.venv/bin/python demo/cli_demo.py --backend onnx \
  --model-dir models/trustlaya-s-v2 \
  --onnx models/exported/v2/trustlaya_s.onnx \
  --text "Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder."
```

The JSON returns nine effective task scores, raw and calibrated model scores, evidence offsets, advisory `model_action`, final policy `action`, `policy_reason`, confidence, abstention, agent and session risk. The local [HTTP API](src/trustlaya/api.py) wraps the same result for `POST /analyze`.

## Bias, security, calibration, and ethical limits

Synthetic templates underrepresent dialects, informal spelling, difficult benign cases, and novel attack styles. Current confidence is categorical sharpness: median 0.999 on synthetic test, while selective action error **rose** with stricter confidence thresholds. It is not a calibrated correctness estimate. The small authored TR/EN/mixed benchmark is a smoke test only. Regex evidence misses many names and account numbers; model PII scores can mislabel dates. Agent and session heuristics are uncalibrated and may miss indirect chains. INT8 and UNO Q are not field validated. See [model health](reports/model_health.md), [limitations](docs/limitations.md), and [security](docs/security.md).
