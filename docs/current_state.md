# Current state before Advanced work

Inventory taken on 2026-09-24 from local commit `a05f133` and public Hugging Face repositories. Development branch: `feature/trustlaya-advanced`. The existing model artifacts are retained as the v1 baseline.

| Item | Observed state |
|---|---|
| Model | `TrustLaya`, 42,138,641 parameters; 19,185,681 trainable in the recorded run; FP32 safetensors 168,571,188 bytes (160.8 MiB). |
| Base | `ytu-ce-cosmos/turkish-medium-bert-uncased`, MIT; 8 BERT layers, hidden size 512, 8 attention heads, 32,000-word vocabulary. |
| Tokenizer | `BertTokenizerFast`, 32,000 vocabulary items, PAD/UNK/CLS/SEP/MASK; tokenizer-related local files ~1,030,456 bytes. Runtime truncates/pads to 96 tokens. |
| Heads | Shared masked mean pooling; one 9-logit binary risk linear head, 4-logit severity head, 4-logit advisory action head. Nine tasks: PII, secret, prompt injection, dangerous instruction, privacy, security, ethics, oversight, data governance. No learned agent head. |
| Training | 10,000 controlled synthetic rows; 150 supervised steps then 80 resumed steps with weak Laya teacher term on 73 unique texts; batch 32; seed 42. Embeddings and first two layers frozen. |
| Inference | Local PyTorch CPU/MPS or ONNX CPU; per-task temperature scaling; PII/secret evidence score floors; separate deterministic YAML policy; action, reason, evidence spans, categorical confidence and abstain flag. |
| Export | FP32 ONNX 167,661,320 bytes; INT8 ONNX 42,507,021 bytes; FP16 safetensors 84,293,722 bytes. ONNX opset 17, inputs `input_ids`/`attention_mask`, outputs `risks`/`severity`/`action`. |
| Dataset | 10,000 synthetic rows, train 6,972, validation 1,053, test 1,975; family-disjoint assignment. Public dataset card labels data as synthetic. Full duplicate and near-duplicate audit remains to be done. |
| Demo | Public Gradio Space runs ONNX CPU with text, nine risk bars, evidence, final action and agent permission checkboxes. UI warns against submitting real PII or secrets. |
| Tests | 11 pytest tests for basic model, tokenizer, PII/secret evidence, policy, inference, calibration and ONNX validity. |

Public resources: [model](https://huggingface.co/ege-arhan/TrustLaya-S), [dataset](https://huggingface.co/datasets/ege-arhan/TrustLaya-S-synthetic), [Space](https://huggingface.co/spaces/ege-arhan/TrustLaya-S-demo), [source](https://github.com/ege-arhan/trustlaya-s). Hugging Face API resolved all three `ege-arhan` repositories and returned 13 model files, 5 dataset files and 6 Space files. Older `xzwq` URLs resolve to the same revisions.

## Recorded baseline, not a new measurement

The existing synthetic test report records mean task accuracy 0.9066 and macro F1 0.6632. Existing batch-one warm p50 latencies are PyTorch CPU 12.388 ms, MPS 6.031 ms, ONNX CPU 5.087 ms. These must be remeasured before Advanced comparisons. Independent diagnostics show prompt-injection false-positive rate 0.429 on deepset held-out examples and model-only secret false-positive rate 0.949 on a separate synthetic/augmented corpus. Synthetic calibration does not establish real-world probabilities.

## Gaps against Advanced objective

No session or cumulative risk, explicit risk-fusion alternatives, learned agent-risk head, ML span head, independent adversarial benchmark, language-stratified generalization benchmark, dataset near-duplicate audit, HTTP API, privacy-preserving audit log, coverage-risk tuning, reliability diagram, Advanced model checkpoint, Tiny model, UNO Q measurement, CI, or unified health/results reports. Current training/export scripts write v1 artifact paths directly; they need versioned output options before any new training or export run.
