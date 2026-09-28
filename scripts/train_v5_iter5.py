"""V5 iteration 5 training (plan: reports/iter5_plan.md). Synthetic rows are TRAIN-only.

Run: .venv/bin/python scripts/train_v5_iter5.py [--only ID ...]
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_tensor_trust import ROOT  # noqa: E402
from train_v5_ablation import TOKENIZER, load_split, tokenize  # noqa: E402
from train_v5_benign_repair import PREDICTIONS, run  # noqa: E402
from train_v5_iter3 import DEV_SOURCES, H1, rows_iter3  # noqa: E402

OUT = ROOT / "reports/experiments/iter5"
CHECKPOINTS = ROOT / "models/v5-iter5"
TR_ATTACK = ("tr_curated_attack", "mt_tr_attack")
DE_ATTACK = ("mt_de_attack",)
BENIGN = ("tr_curated_benign", "mt_de_benign", "mt_tr_benign")
RUNS = {
    "S1_tr": TR_ATTACK, "S2_de": DE_ATTACK, "S3_tr_de": TR_ATTACK + DE_ATTACK,
    "S4_tr_de_benign": TR_ATTACK + DE_ATTACK + BENIGN, "S5_de_debenign": DE_ATTACK + ("mt_de_benign",),
}


def rows_iter5():
    synthetic = [json.loads(line) for line in open(ROOT / "data/v5_iter5/synthetic_rows.jsonl")]
    for r in synthetic:  # split tr_curated by label so attack and benign can be added separately
        if r["source"] == "tr_curated":
            r["source"] = "tr_curated_attack" if r["label"] == "ATTACK" else "tr_curated_benign"
    return rows_iter3() + synthetic


def spec(run_id, seed=42):
    return {**H1, "batching": "grouped", "seed": seed, "synthetic_sources": RUNS[run_id]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    rows = rows_iter5()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", DEV_SOURCES))
    assert not any(r.get("synthetic") for r in dev_rows), "synthetic rows must never be in DEV"
    for run_id in args.only or RUNS:
        name = f"{run_id}_s{args.seed}"
        if (OUT / name / "checkpoint.sha256").exists():
            print(f"{name}: already trained, skipping", flush=True)
            continue
        run(name, rows, tokenizer, dev_rows, spec=spec(run_id, args.seed), out_root=OUT, ckpt_root=CHECKPOINTS)


if __name__ == "__main__":
    main()
