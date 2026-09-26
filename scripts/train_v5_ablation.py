"""V5 training ablation (phase 1: train + DEV only). TEST sets are never loaded here.

Every experiment: same seed, optimizer, schedule and epoch budget. The epoch
kept is the one with the lowest DEV loss on the shared DEV set. Checkpoints go
to git-ignored models/v5-ablation/<id>/; their SHA-256 and all configs, data
manifests, logs and DEV metrics go to reports/experiments/<id>/.

Selection rules below are written before any run and are not changed after.
Run: .venv/bin/python scripts/train_v5_ablation.py [--only ID ...]
"""

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v5_evidence import metrics  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from trustlaya.utils import device, normalize  # noqa: E402
from trustlaya.v5_model import V5Classifier, batch, focal_loss, source_balanced_weights  # noqa: E402

BACKBONE = ROOT / "models/base"
TOKENIZER = ROOT / "models/trustlaya-s-v2"  # same vocabulary as the base model; V2 normalization
CHECKPOINTS = ROOT / "models/v5-ablation"
EXPERIMENTS_DIR = ROOT / "reports/experiments"
PREDICTIONS = ROOT / "data/v5/predictions"
SEED = 42
TRAINING = {"optimizer": "AdamW", "lr": 2e-5, "weight_decay": 0.01, "batch_size": 16,
            "epochs": 3, "warmup_fraction": 0.1, "schedule": "linear warmup + linear decay",
            "grad_clip": 1.0, "init": "models/base (ytu-ce-cosmos/turkish-medium-bert-uncased), not V2",
            "precision": "fp32", "reading": "HEAD truncation", "epoch_selection": "lowest DEV loss (shared DEV)"}
SELECTION_RULES = {
    "threshold": "per experiment, on DEV only: maximize DEV macro group accuracy over the (source,label) "
                 "groups TT/ATTACK, JLL/ATTACK, JLL/BENIGN, security_docs/BENIGN, arxiv/BENIGN; grid 0.01..0.99",
    "candidate": "highest DEV macro group accuracy at its DEV threshold; tie-break DEV PR-AUC",
    "calibration": "temperature scaling fitted on DEV only for every experiment; raw and calibrated reported",
    "test": "TEST1 (pool TEST) and TEST2 (OOD) run once, after selection is frozen, for all experiments",
}
MIXED = ("tensor_trust", "jailbreakllms", "security_docs", "arxiv_abstracts")
EXPERIMENTS = {
    "E1_tt_ce_94": {"sources": ("tensor_trust", "security_docs", "arxiv_abstracts"),
                    "sampling": "NATURAL", "loss": "CE", "context": 94,
                    "note": "Tensor Trust has no benign rows; negatives are public security prose only."},
    "E2_jll_ce_94": {"sources": ("jailbreakllms",), "sampling": "NATURAL", "loss": "CE", "context": 94},
    "E3_mixed_natural_ce_94": {"sources": MIXED, "sampling": "NATURAL", "loss": "CE", "context": 94},
    "E4_mixed_balanced_ce_94": {"sources": MIXED, "sampling": "BALANCED_SOURCE", "loss": "CE", "context": 94},
    "E5_mixed_natural_focal_510": {"sources": MIXED, "sampling": "NATURAL", "loss": "FOCAL", "context": 510},
    "E6_mixed_balanced_focal_510": {"sources": MIXED, "sampling": "BALANCED_SOURCE", "loss": "FOCAL", "context": 510},
    "E7_mixed_balanced_ce_256": {"sources": MIXED, "sampling": "BALANCED_SOURCE", "loss": "CE", "context": 256},
    "E8_mixed_balanced_ce_510": {"sources": MIXED, "sampling": "BALANCED_SOURCE", "loss": "CE", "context": 510},
}
FOCAL = {"gamma": 2.0, "alpha": 0.25}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_split(rows, split, sources=None):
    return [r for r in rows if r["status"] == "included" and r["split"] == split
            and r["label"] in ("ATTACK", "BENIGN") and (sources is None or r["source"] in sources)]


def tokenize(tokenizer, rows):
    for r in rows:
        if "tokens" not in r:
            r["tokens"] = tokenizer(normalize(r["text"]), add_special_tokens=False, verbose=False)["input_ids"]
            r["y"] = float(r["label"] == "ATTACK")
    return rows


def epoch_order(n, weights, epoch):
    generator = torch.Generator().manual_seed(SEED + epoch)
    if weights is None:
        return torch.randperm(n, generator=generator).tolist()
    return torch.multinomial(torch.tensor(weights, dtype=torch.double), n, replacement=True,
                             generator=generator).tolist()


def predict(model, rows, context, tokenizer, dev):
    model.eval()
    probs, losses = [], []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tokens"]))
    out = [0.0] * len(rows)
    with torch.inference_mode():
        for start in range(0, len(order), 64):
            idx = order[start:start + 64]
            ids, mask = batch([rows[i]["tokens"] for i in idx], context, tokenizer.cls_token_id,
                              tokenizer.sep_token_id)
            logits = model(ids.to(dev), mask.to(dev)).float().cpu()
            targets = torch.tensor([rows[i]["y"] for i in idx])
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="sum").item())
            for i, value in zip(idx, torch.sigmoid(logits).tolist()):
                out[i] = value
    return out, sum(losses) / len(rows)


def environment():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": np.__version__, "device": str(device()),
            "git_commit": commit, "note": "MPS kernels are not bitwise deterministic; seeds fix data order and init."}


def run(experiment_id, rows, tokenizer, dev_rows):
    spec = EXPERIMENTS[experiment_id]
    out_dir = EXPERIMENTS_DIR / experiment_id
    ckpt_dir = CHECKPOINTS / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train = tokenize(tokenizer, load_split(rows, "TRAIN", set(spec["sources"])))
    groups = [(r["source"], r["label"]) for r in train]
    weights = source_balanced_weights(groups) if spec["sampling"] == "BALANCED_SOURCE" else None
    counts = Counter(f"{s}/{l}" for s, l in groups)
    ids_digest = hashlib.sha256("\n".join(sorted(r["sample_id"] for r in train)).encode()).hexdigest()
    expected_mass = {}
    for g, w in zip(groups, weights or [1 / len(train)] * len(train)):
        expected_mass[f"{g[0]}/{g[1]}"] = expected_mass.get(f"{g[0]}/{g[1]}", 0) + w
    kish = (sum(weights) ** 2 / sum(w * w for w in weights)) if weights else len(train)
    config = {"experiment_id": experiment_id, **spec, "sources": list(spec["sources"]), "seed": SEED,
              "training": TRAINING, "focal": FOCAL if spec["loss"] == "FOCAL" else None,
              "selection_rules": SELECTION_RULES, "tokenizer_sha256": sha256(TOKENIZER / "tokenizer.json"),
              "backbone_sha256": sha256(BACKBONE / "model.safetensors")}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokens = [len(r["tokens"]) for r in train]
    words = [max(1, len(r["text"].split())) for r in train]
    manifest = {"train_rows": len(train), "train_by_source_label": dict(sorted(counts.items())),
                "class_balance": {"ATTACK": sum(r["y"] for r in train), "BENIGN": len(train) - sum(r["y"] for r in train)},
                "sampling": spec["sampling"], "samples_per_epoch": len(train),
                "expected_sampling_mass": {k: round(v / sum(expected_mass.values()), 4) for k, v in expected_mass.items()},
                "kish_effective_sample_size": round(kish, 1),
                "train_sample_ids_sha256": ids_digest,
                "dev_rows": len(dev_rows), "dev_sample_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(r["sample_id"] for r in dev_rows)).encode()).hexdigest(),
                "v5_manifest_sha256": (ROOT / "data/v5_manifest.sha256").read_text(),
                "tokenizer_stats": {"language": "English-dominant; 0 Turkish rows in these sources",
                                    "median_tokens": float(np.median(tokens)),
                                    "expansion_ratio": round(sum(tokens) / sum(words), 3),
                                    "truncation_rate_at_context": round(sum(t > spec["context"] for t in tokens) / len(tokens), 4)}}

    seed_everything(SEED)
    dev = device()
    model = V5Classifier(BACKBONE).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
    steps_per_epoch = math.ceil(len(train) / TRAINING["batch_size"])
    total = steps_per_epoch * TRAINING["epochs"]
    warmup = int(total * TRAINING["warmup_fraction"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.0, (total - s) / max(1, total - warmup)))
    log = {"steps_per_epoch": steps_per_epoch, "epochs": [], "train_loss_every_50_steps": []}
    best = None
    seen = set()
    started = time.perf_counter()
    for epoch in range(TRAINING["epochs"]):
        model.train()
        order = epoch_order(len(train), weights, epoch)
        seen.update(order)
        running, epoch_loss = [], 0.0
        for step in range(steps_per_epoch):
            idx = order[step * TRAINING["batch_size"]:(step + 1) * TRAINING["batch_size"]]
            ids, mask = batch([train[i]["tokens"] for i in idx], spec["context"],
                              tokenizer.cls_token_id, tokenizer.sep_token_id)
            targets = torch.tensor([train[i]["y"] for i in idx], device=dev)
            logits = model(ids.to(dev), mask.to(dev))
            loss = (focal_loss(logits, targets, **FOCAL) if spec["loss"] == "FOCAL" else
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["grad_clip"])
            optimizer.step()
            scheduler.step()
            running.append(loss.item())
            epoch_loss += loss.item()
            if len(running) == 50:
                log["train_loss_every_50_steps"].append(round(sum(running) / 50, 5))
                running = []
        probs, dev_loss = predict(model, dev_rows, spec["context"], tokenizer, dev)
        entry = {"epoch": epoch + 1, "train_loss": round(epoch_loss / steps_per_epoch, 5),
                 "dev_bce_loss": round(dev_loss, 5), "elapsed_s": round(time.perf_counter() - started, 1)}
        log["epochs"].append(entry)
        print(f"{experiment_id} epoch {epoch + 1}: {entry}", flush=True)
        if best is None or dev_loss < best["dev_bce_loss"]:
            best = {**entry}
            model.save(ckpt_dir / "model.safetensors")
            np.save(PREDICTIONS / f"{experiment_id}_DEV.npy", np.asarray(probs))
    log.update(best_epoch=best["epoch"], wall_clock_training_s=round(time.perf_counter() - started, 1),
               distinct_training_rows_seen=len(seen))
    manifest["distinct_training_rows_seen"] = len(seen)
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
    (out_dir / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    ckpt = ckpt_dir / "model.safetensors"
    (out_dir / "checkpoint.sha256").write_text(f"{sha256(ckpt)}  models/v5-ablation/{experiment_id}/model.safetensors\n")
    probs = np.load(PREDICTIONS / f"{experiment_id}_DEV.npy")
    y = [r["y"] for r in dev_rows]
    metrics_out = {"parameters": sum(p.numel() for p in model.parameters()),
                   "checkpoint_bytes": ckpt.stat().st_size, "best_epoch": best["epoch"],
                   "DEV_threshold_0.5": {"pooled": metrics(y, probs),
                                         "per_source": per_source(dev_rows, probs, 0.5)}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2) + "\n")


def per_source(rows, probs, threshold):
    out = {}
    for source in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == source]
        out[source] = metrics([rows[i]["y"] for i in idx], [probs[i] for i in idx], threshold)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(EXPERIMENTS))
    args = parser.parse_args()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(ROOT / "data/v5/all_rows.jsonl")]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", set(MIXED)))
    for experiment_id in args.only or EXPERIMENTS:
        if (EXPERIMENTS_DIR / experiment_id / "checkpoint.sha256").exists():
            print(f"{experiment_id}: already trained, skipping (delete its report dir to retrain)")
            continue
        run(experiment_id, rows, tokenizer, dev_rows)


if __name__ == "__main__":
    main()
