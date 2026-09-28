"""V5 iteration 3 training (plan: reports/iter3_plan.md). DEV only; TEST sets are not loaded.

Run: .venv/bin/python scripts/train_v5_iter3.py [--only ID ...]
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_tensor_trust import ROOT  # noqa: E402
from train_v5_ablation import MIXED, TOKENIZER, load_split, tokenize  # noqa: E402
from train_v5_benign_repair import ALL, NEW_SOURCES, PREDICTIONS, load_rows, run  # noqa: E402

OUT = ROOT / "reports/experiments/iter3"
CHECKPOINTS = ROOT / "models/v5-iter3"
H1 = {"languages": ALL, "sampling": "BALANCED_SOURCE", "new_weight": 4, "extra_attack_sources": ("hackaprompt",)}
RUNS = {
    "H1_grouped_s42": {**H1, "batching": "grouped", "seed": 42},
    "H1_grouped_s43": {**H1, "batching": "grouped", "seed": 43},
    "H1_grouped_s44": {**H1, "batching": "grouped", "seed": 44},
    "H1_random_s42": {**H1, "batching": "random", "seed": 42},
}
DEV_SOURCES = set(MIXED) | set(NEW_SOURCES) | {"hackaprompt"}


def rows_iter3():
    return load_rows() + [json.loads(line) for line in open(ROOT / "data/v5_iter3/new_rows.jsonl")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(RUNS))
    args = parser.parse_args()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    rows = rows_iter3()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", DEV_SOURCES))
    for run_id in args.only or RUNS:
        if (OUT / run_id / "checkpoint.sha256").exists():
            print(f"{run_id}: already trained, skipping", flush=True)
            continue
        run(run_id, rows, tokenizer, dev_rows, spec=RUNS[run_id], out_root=OUT, ckpt_root=CHECKPOINTS)


if __name__ == "__main__":
    main()
