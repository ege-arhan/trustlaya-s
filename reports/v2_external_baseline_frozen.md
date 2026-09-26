# TrustLaya-S v2 external baseline — frozen before v3 work

Rechecked on 2026-09-25 from repository commit `4a1e78e`. The unchanged
v2 checkpoint was evaluated with `PYTHONPATH=src .venv/bin/python
scripts/run_external_real.py` before any v3 training or threshold selection.
The previously published [v2 external report](external_real_world_benchmark.md)
and [v2 results](external_results.json) remain unchanged in Git.

| Frozen task | Test source | N | TP | FP | TN | FN | Precision | Recall | F1 | FPR | FNR | ROC-AUC | PR-AUC | ECE | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Narrow DIRECT PERSON/CODE presence per 94-token window | TAB official test, 127 ECHR cases | 2,079 | 21 | 154 | 1,706 | 198 | 0.120 | 0.096 | 0.107 | 0.083 | 0.904 | 0.593 | 0.125 | 0.148 | 0.147 |
| Direct jailbreak proxy through prompt-injection head | JailbreakLLMs community subset | 5,888 | 601 | 4,726 | 526 | 35 | 0.113 | 0.945 | 0.202 | 0.900 | 0.055 | 0.707 | 0.249 | 0.736 | 0.677 |

Raw-score decision threshold: **0.50** for both. TAB is a binary window
presence projection of annotated spans; the v2 classifier does not predict
entities. Its regex evidence exact-span recall was 0/351 on this mapping.
The jailbreak set has 24 conflicting-label rows and 262 duplicates removed;
213 prompt-repository rows were excluded. Training-data exact and char-ngram
near overlap checks found no retained external rows in the recorded v2 training
sources. This cannot exclude semantic or encoder-pretraining contamination.

## Immutable artifact references

| Artifact | SHA-256 / revision |
|---|---|
| v2 `model.safetensors` | `99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c` |
| v2 FP32 ONNX | `9d3b953709d990ef84b71c46f63dadec2076ca5b9f01cfd7c7d4ddc8e170526f` |
| TAB repository | `558e09e26d6b36f5f78440074e6a233946d98bd9` |
| JailbreakLLMs repository | `2dbd7bbc25f1b156552678f451bddbc787cd679f` |
| Local frozen TAB predictions | `06230d6aa26c5130045b21f640ee8dd4f85e6b40b07ca0515468810b1e7c4ab2` |
| Local frozen jailbreak predictions | `bd2454d22ecba54fef0e245ef207a7d65f9029f73fecbe9af6fc236356a5224b` |
| Local rerun aggregate JSON | `300708a6bf3f74969b845b561f3ef663777831cb05aa4fdc59a60e136d97f7b9` |

The prediction files are locally preserved in the gitignored
`benchmarks/external/predictions/v2_frozen_*.json`; they contain hashes, IDs,
scores and evidence offsets, without raw text. The rerun's aggregate JSON
differs from the committed v2 JSON only in runtime latency observations.

## Rerun environment

macOS 26.3.1, Apple arm64, Python 3.12.10, 24 GiB RAM, MPS available.
PyTorch 2.14.0, Transformers 4.57.6, Datasets 5.0.1, ONNX Runtime 1.30.0,
tokenizers 0.22.2, safetensors 0.8.0. The frozen evaluation used ONNX CPU;
MPS availability does not imply MPS was used for these scores.
