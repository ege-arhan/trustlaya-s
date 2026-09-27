"""V5 iteration 2 (benign repair), phase 1: train + DEV only. Plan: reports/benign_repair_plan.md.

Attack TRAIN is identical to E6 in every run; only benign TRAIN and sampling change.
TEST splits, deepset and MASSIVE are never loaded here. Outputs go to
reports/experiments/benign_repair/<id>/ and git-ignored models/v5-benign-repair/<id>/.

Run: .venv/bin/python scripts/train_v5_benign_repair.py [--only ID ...]
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v5_evidence import metrics  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from train_v5_ablation import (BACKBONE, FOCAL, MIXED, SEED, TOKENIZER, TRAINING, environment, epoch_order,  # noqa: E402
                               load_split, predict, seed_everything, tokenize)
from trustlaya.utils import device  # noqa: E402
from trustlaya.v5_model import V5Classifier, batch, focal_loss, source_balanced_weights  # noqa: E402

OUT = ROOT / "reports/experiments/benign_repair"
CHECKPOINTS = ROOT / "models/v5-benign-repair"
PREDICTIONS = ROOT / "data/v5_benign/predictions"
NEW_SOURCES = ("dolly", "oasst2", "aya")
ALL = ("en", "de", "tr")
CONTEXT = 510
CHUNK_BATCHES = 50
RUNS = {
    "R0_control": {"languages": (), "sampling": "BALANCED_SOURCE", "new_weight": 1},
    "B1_en": {"languages": ("en",), "sampling": "BALANCED_SOURCE", "new_weight": 1},
    "B2_de": {"languages": ("de",), "sampling": "BALANCED_SOURCE", "new_weight": 1},
    "B3_tr": {"languages": ("tr",), "sampling": "BALANCED_SOURCE", "new_weight": 1},
    "B4_multilingual": {"languages": ALL, "sampling": "BALANCED_SOURCE", "new_weight": 1},
    "M1_multilingual_natural": {"languages": ALL, "sampling": "NATURAL", "new_weight": 1},
    "A2_multilingual_2x": {"languages": ALL, "sampling": "BALANCED_SOURCE", "new_weight": 2},
    "A4_multilingual_4x": {"languages": ALL, "sampling": "BALANCED_SOURCE", "new_weight": 4},
}


def load_rows():
    old = [json.loads(line) for line in open(ROOT / "data/v5/all_rows.jsonl")]
    for r in old:
        r.setdefault("language", "unknown")
    new = [json.loads(line) for line in open(ROOT / "data/v5_benign/new_rows.jsonl")]
    return old + new


def weights_for(train, spec):
    if spec["sampling"] == "NATURAL":
        return None
    weights = source_balanced_weights([(r["source"], r["label"]) for r in train])
    if spec["new_weight"] != 1:
        weights = [w * spec["new_weight"] if r["source"] in NEW_SOURCES else w for r, w in zip(train, weights)]
        benign = sum(w for r, w in zip(train, weights) if r["label"] == "BENIGN")
        attack = sum(w for r, w in zip(train, weights) if r["label"] == "ATTACK")
        weights = [w * attack / benign if r["label"] == "BENIGN" else w for r, w in zip(train, weights)]
    return weights


def grouped_batches(order, lengths, batch_size, epoch):
    """Sort by length inside chunks of CHUNK_BATCHES batches, then shuffle batch order (seeded)."""
    batches = []
    size = batch_size * CHUNK_BATCHES
    for start in range(0, len(order), size):
        chunk = sorted(order[start:start + size], key=lambda i: lengths[i])
        batches += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
    random.Random(SEED + 1000 + epoch).shuffle(batches)
    return batches


def ids_digest(rows):
    return hashlib.sha256("\n".join(sorted(r["sample_id"] for r in rows)).encode()).hexdigest()


def run(run_id, rows, tokenizer, dev_rows):
    spec = RUNS[run_id]
    out_dir, ckpt_dir = OUT / run_id, CHECKPOINTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base = load_split(rows, "TRAIN", set(MIXED))
    extra = [r for r in load_split(rows, "TRAIN", set(NEW_SOURCES)) if r["language"] in spec["languages"]]
    train = tokenize(tokenizer, base + extra)
    weights = weights_for(train, spec)
    lengths = [min(len(r["tokens"]), CONTEXT) for r in train]
    mass = Counter()
    for r, w in zip(train, weights or [1 / len(train)] * len(train)):
        mass[f"{r['source']}/{r['label']}"] += w
    total = sum(mass.values())
    kish = (sum(weights) ** 2 / sum(w * w for w in weights)) if weights else len(train)
    config = {"run_id": run_id, **spec, "languages": list(spec["languages"]), "context": CONTEXT, "loss": "FOCAL",
              "focal": FOCAL, "seed": SEED, "training": {**TRAINING, "batching": f"length-grouped, chunks of {CHUNK_BATCHES} batches"},
              "plan": "reports/benign_repair_plan.md", "tokenizer_sha256": sha256(TOKENIZER / "tokenizer.json"),
              "backbone_sha256": sha256(BACKBONE / "model.safetensors")}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"train_rows": len(train),
                "train_by_source_language_label": dict(sorted(Counter(
                    f"{r['source']}/{r.get('language', 'unknown')}/{r['label']}" for r in train).items())),
                "class_balance": dict(Counter(r["label"] for r in train)),
                "expected_sampling_mass": {k: round(v / total, 4) for k, v in sorted(mass.items())},
                "kish_effective_sample_size": round(kish, 1), "samples_per_epoch": len(train),
                "new_benign_length_buckets": dict(Counter(
                    next(n for n, lo, hi in (("<=32", 0, 32), ("33-64", 33, 64), ("65-94", 65, 94),
                                              ("95-256", 95, 256), ("257+", 257, 10 ** 9)) if lo <= len(r["tokens"]) <= hi)
                    for r in extra))}
    checksums = {"train_sample_ids_sha256": ids_digest(train), "dev_sample_ids_sha256": ids_digest(dev_rows),
                 "v5_manifest.sha256": (ROOT / "data/v5_manifest.sha256").read_text(),
                 "benign_repair_manifest.sha256": (ROOT / "data/benign_repair_manifest.sha256").read_text(),
                 "config_sha256": sha256(out_dir / "config.json")}
    (out_dir / "dataset_checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")

    seed_everything(SEED)
    dev = device()
    model = V5Classifier(BACKBONE).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
    steps_per_epoch = math.ceil(len(train) / TRAINING["batch_size"])
    total_steps = steps_per_epoch * TRAINING["epochs"]
    warmup = int(total_steps * TRAINING["warmup_fraction"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup)) * max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    log = {"steps_per_epoch": steps_per_epoch, "epochs": [], "train_loss_every_50_steps": []}
    best, seen, started = None, set(), time.perf_counter()
    for epoch in range(TRAINING["epochs"]):
        model.train()
        order = epoch_order(len(train), weights, epoch)
        seen.update(order)
        running, epoch_loss, steps = [], 0.0, 0
        for idx in grouped_batches(order, lengths, TRAINING["batch_size"], epoch):
            ids, mask = batch([train[i]["tokens"] for i in idx], CONTEXT, tokenizer.cls_token_id, tokenizer.sep_token_id)
            targets = torch.tensor([train[i]["y"] for i in idx], device=dev)
            loss = focal_loss(model(ids.to(dev), mask.to(dev)), targets, **FOCAL)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING["grad_clip"])
            optimizer.step()
            scheduler.step()
            running.append(loss.item()); epoch_loss += loss.item(); steps += 1
            if len(running) == 50:
                log["train_loss_every_50_steps"].append(round(sum(running) / 50, 5)); running = []
        probs, dev_loss = predict(model, dev_rows, CONTEXT, tokenizer, dev)
        entry = {"epoch": epoch + 1, "train_loss": round(epoch_loss / steps, 5), "dev_bce_loss": round(dev_loss, 5),
                 "elapsed_s": round(time.perf_counter() - started, 1)}
        log["epochs"].append(entry)
        print(f"{run_id} epoch {epoch + 1}: {entry}", flush=True)
        if best is None or dev_loss < best["dev_bce_loss"]:
            best = dict(entry)
            model.save(ckpt_dir / "model.safetensors")
            np.save(PREDICTIONS / f"{run_id}_DEV.npy", np.asarray(probs))
    log.update(best_epoch=best["epoch"], wall_clock_training_s=round(time.perf_counter() - started, 1),
               distinct_training_rows_seen=len(seen))
    manifest["distinct_training_rows_seen"] = len(seen)
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
    (out_dir / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    ckpt = ckpt_dir / "model.safetensors"
    (out_dir / "checkpoint.sha256").write_text(f"{sha256(ckpt)}  models/v5-benign-repair/{run_id}/model.safetensors\n")
    probs = np.load(PREDICTIONS / f"{run_id}_DEV.npy")
    y = [r["y"] for r in dev_rows]
    (out_dir / "metrics_dev.json").write_text(json.dumps({
        "parameters": sum(p.numel() for p in model.parameters()), "checkpoint_bytes": ckpt.stat().st_size,
        "best_epoch": best["epoch"], "DEV_threshold_0.5": metrics(y, probs)}, indent=2) + "\n")
    per_source = {}
    for source in sorted({r["source"] for r in dev_rows}):
        idx = [i for i, r in enumerate(dev_rows) if r["source"] == source]
        per_source[source] = metrics([y[i] for i in idx], probs[idx])
    (out_dir / "source_metrics.json").write_text(json.dumps({"DEV_threshold_0.5": per_source}, indent=2) + "\n")


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
        run(run_id, rows, tokenizer, dev_rows)


if __name__ == "__main__":
    main()
