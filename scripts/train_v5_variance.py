"""V5 variance study: seed and batching effects. Descriptive only; no model is selected here.

E6 data = V5 v1 TRAIN (no new benign); A4 data = + en/de/tr benign at 4x weight.
"random" batching reproduces E6's recipe; "grouped" is the iteration-2 recipe.
Existing seed-42 runs (E6, R0_control, A4_multilingual_4x) are reused, not retrained.
Epochs are chosen by the shared iteration-2 DEV loss, as in iteration 2.

Run: .venv/bin/python scripts/train_v5_variance.py [--only ID ...]
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_tensor_trust import ROOT  # noqa: E402
from train_v5_ablation import MIXED, TOKENIZER, load_split, tokenize  # noqa: E402
from train_v5_benign_repair import ALL, NEW_SOURCES, PREDICTIONS, load_rows, run  # noqa: E402

OUT = ROOT / "reports/experiments/variance"
CHECKPOINTS = ROOT / "models/v5-variance"
E6_DATA = {"languages": (), "sampling": "BALANCED_SOURCE", "new_weight": 1}
A4_DATA = {"languages": ALL, "sampling": "BALANCED_SOURCE", "new_weight": 4}
RUNS = {
    "E6data_random_s43": {**E6_DATA, "batching": "random", "seed": 43},
    "E6data_random_s44": {**E6_DATA, "batching": "random", "seed": 44},
    "E6data_grouped_s43": {**E6_DATA, "batching": "grouped", "seed": 43},
    "E6data_grouped_s44": {**E6_DATA, "batching": "grouped", "seed": 44},
    "A4data_grouped_s43": {**A4_DATA, "batching": "grouped", "seed": 43},
    "A4data_grouped_s44": {**A4_DATA, "batching": "grouped", "seed": 44},
    "A4data_random_s42": {**A4_DATA, "batching": "random", "seed": 42},
}
REUSED = {"E6data_random_s42": "E6_mixed_balanced_focal_510 (reports/experiments)",
          "E6data_grouped_s42": "R0_control (reports/experiments/benign_repair)",
          "A4data_grouped_s42": "A4_multilingual_4x (reports/experiments/benign_repair)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(RUNS))
    args = parser.parse_args()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", set(MIXED) | set(NEW_SOURCES)))
    for run_id in args.only or RUNS:
        if (OUT / run_id / "checkpoint.sha256").exists():
            print(f"{run_id}: already trained, skipping", flush=True)
            continue
        run(run_id, rows, tokenizer, dev_rows, spec=RUNS[run_id], out_root=OUT, ckpt_root=CHECKPOINTS)


if __name__ == "__main__":
    main()
