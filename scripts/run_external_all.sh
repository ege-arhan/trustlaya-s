#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f models/trustlaya-s-v2/model.safetensors ] || [ ! -f models/exported/v2/trustlaya_s.onnx ]; then
  .venv/bin/python scripts/download_artifacts.py --advanced
fi
if [ ! -f data/splits/train.jsonl ]; then
  PYTHONPATH=src .venv/bin/python scripts/build_dataset.py --count 10000
fi
mkdir -p benchmarks/external/raw
if [ ! -d benchmarks/external/raw/tab/.git ]; then
  git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark.git benchmarks/external/raw/tab
fi
git -C benchmarks/external/raw/tab checkout --detach 558e09e26d6b36f5f78440074e6a233946d98bd9
if [ ! -d benchmarks/external/raw/jailbreakllms/.git ]; then
  git clone https://github.com/TrustAIRLab/JailbreakLLMs.git benchmarks/external/raw/jailbreakllms
fi
git -C benchmarks/external/raw/jailbreakllms checkout --detach 2dbd7bbc25f1b156552678f451bddbc787cd679f
if [ ! -d benchmarks/external/raw/glaucoma/.git ]; then
  git clone https://github.com/jche253/Glaucoma_Med_Dataset.git benchmarks/external/raw/glaucoma
fi
git -C benchmarks/external/raw/glaucoma checkout --detach 15e67ea86ed0678425ec0948032f3fe9f77f30db
uv pip install --python .venv/bin/python -r requirements-external.txt
PYTHONPATH=src .venv/bin/python scripts/run_external_real.py
PYTHONPATH=src .venv/bin/python scripts/run_external_baselines.py
PYTHONPATH=src .venv/bin/python scripts/run_neuraltrust_baseline.py
PYTHONPATH=src .venv/bin/python scripts/audit_jailbreak_baseline.py
PYTHONPATH=src .venv/bin/python scripts/calibrate_external_tab.py
PYTHONPATH=src .venv/bin/python scripts/run_external_ablation.py
PYTHONPATH=src .venv/bin/python scripts/run_clinical_unlabeled.py
PYTHONPATH=src .venv/bin/python scripts/write_external_report.py
