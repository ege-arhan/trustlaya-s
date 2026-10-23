"""Train frozen-encoder v3 token and attack heads from external TRAIN only.

Run: PYTHONPATH=src .venv/bin/python scripts/train_v3_external.py
Final tests are deliberately not opened by this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoTokenizer

from trustlaya.model import TrustLaya, BACKBONE
from trustlaya.utils import normalize
from trustlaya.v3_external import BIO, binary_metrics, calibration_metrics, fingerprint, token_labels
from run_external_real import tab_samples

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks/external/raw/v3"
OUT = ROOT / "models/trustlaya-s-v3"
SEED = 20260925
MAXLEN = 96
V2_SHA256 = "99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c"
PROMPTS_SHA256 = "c506bbf29106058a021e5cf85271bb97c9856c2b7fcc9f337421cdc8b00964c6"
GANDALF_SHA256 = {
    "train": "5b6acf3e5a5998d21f8e1222bb45bbdec25a14408747b1cd63bebef4a75fa439",
    "validation": "f51ab3e3407a368845b0f57932cc745c09280429416d7507bd178a16326a79f6",
    "test": "56b646d133335ebc535266bd55dbe1b5bee7caa4b95bf49d040684b9b5dd9972",
}


def gandalf(split):
    path = next((RAW / "gandalf/data").glob(f"{split}-*.parquet"))
    if split in GANDALF_SHA256:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == GANDALF_SHA256[split], f"Unexpected Gandalf {split} data"
    return [str(r["text"]).strip() for r in pq.read_table(path).to_pylist() if str(r["text"]).strip()]


ATTACK_DIRECTIVE = re.compile(
    r"\b(?:jailbreak|DAN\b|do anything now|bypass (?:safety|rules|policy)|ignore (?:all |any |the |your )?(?:previous|prior|above|system|safety)|reveal (?:your |the )?(?:system prompt|password|secret)|disable (?:your )?(?:safety|guardrails)|override (?:system|developer|instructions))",
    re.I,
)


def benign_prompts():
    assert hashlib.sha256((RAW / "prompts.csv").read_bytes()).hexdigest() == PROMPTS_SHA256, "Unexpected prompts.chat data"
    csv.field_size_limit(sys.maxsize)
    with (RAW / "prompts.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {"train": [], "validation": []}
    seen = set(); reasons = {"total": len(rows), "nontext": 0, "length": 0, "attack_directive": 0, "duplicate": 0}
    for r in rows:
        t = (r["prompt"] or "").strip()
        if r["type"] != "TEXT": reasons["nontext"] += 1; continue
        if not 20 <= len(t) <= 5000: reasons["length"] += 1; continue
        if ATTACK_DIRECTIVE.search(t): reasons["attack_directive"] += 1; continue
        h = fingerprint(t)
        if h in seen: reasons["duplicate"] += 1; continue
        seen.add(h)
        contributor = (r["contributor"] or r["act"] or h).strip()
        bucket = int(hashlib.sha256(contributor.encode()).hexdigest()[:8], 16) % 5
        result["validation" if bucket == 0 else "train"].append(t)
    reasons.update(train=len(result["train"]), validation=len(result["validation"]))
    return result, reasons


def encode(model, tokenizer, texts, device, token_features=False, batch_size=24):
    out = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer([normalize(t) for t in texts[start:start + batch_size]],
                              padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            feat = hidden if token_features else (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            out.append(feat.cpu().numpy().astype(np.float16))
            if start and start % 1200 == 0: print("encoded", start, "/", len(texts), flush=True)
    return np.concatenate(out) if out else np.empty((0, model.encoder.config.hidden_size), np.float16)


def pii_arrays(rows, tokenizer):
    labels = []; valid = []; offsets = []
    for r in rows:
        enc = tokenizer(normalize(r["text"]), truncation=True, max_length=MAXLEN,
                        padding="max_length", return_offsets_mapping=True)
        off = [tuple(x) for x in enc["offset_mapping"]]
        lab = token_labels(off, r["gold_spans"])
        labels.append(lab); valid.append([int(b > a) for a, b in off]); offsets.append(off)
    return np.asarray(labels, dtype=np.int64), np.asarray(valid, dtype=bool), offsets


def train_pii(features, labels, valid, dev_features, dev_labels, dev_valid):
    torch.manual_seed(SEED)
    head = nn.Linear(features.shape[-1], len(BIO))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.01)
    weights = torch.tensor([0.25, 4., 2., 4., 2.])
    x = torch.from_numpy(features.astype(np.float32)); y = torch.from_numpy(labels.copy())
    y[~torch.from_numpy(valid)] = -100
    dx = torch.from_numpy(dev_features.astype(np.float32)); dy = torch.from_numpy(dev_labels.copy()); dy[~torch.from_numpy(dev_valid)] = -100
    history = []
    for epoch in range(1, 9):
        head.train(); order = torch.randperm(len(x)); losses = []
        for indices in order.split(128):
            logits = head(x[indices]); loss = nn.functional.cross_entropy(logits.reshape(-1, 5), y[indices].reshape(-1), weight=weights, ignore_index=-100)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        head.eval()
        with torch.inference_mode():
            pred = head(dx).argmax(-1)
            mask = torch.from_numpy(dev_valid)
            positive_gold = (dy > 0) & mask; positive_pred = (pred > 0) & mask
            tp = int((positive_gold & positive_pred).sum()); fp = int((~positive_gold & positive_pred).sum()); fn = int((positive_gold & ~positive_pred).sum())
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_token_presence_f1": f1})
        print("PII epoch", epoch, history[-1], flush=True)
    return head.eval(), history


def train_attack(x, y, dx, dy):
    torch.manual_seed(SEED + 1)
    head = nn.Linear(x.shape[-1], 1)
    opt = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.01)
    xx = torch.from_numpy(x.astype(np.float32)); yy = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
    dxx = torch.from_numpy(dx.astype(np.float32))
    history = []
    for epoch in range(1, 16):
        head.train(); order = torch.randperm(len(x)); losses = []
        for indices in order.split(128):
            logits = head(xx[indices]); loss = nn.functional.binary_cross_entropy_with_logits(logits, yy[indices], pos_weight=torch.tensor([len(y) / max(1, y.sum()) - 1]))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        head.eval()
        with torch.inference_mode(): score = torch.sigmoid(head(dxx)).flatten().numpy()
        met = binary_metrics(dy, score, .5)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_f1_at_0.5": met["f1"], "dev_fpr_at_0.5": met["fpr"]})
        print("Attack epoch", epoch, history[-1], flush=True)
    return head.eval(), history


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v2")
    assert hashlib.sha256((ROOT / "models/trustlaya-s-v2/model.safetensors").read_bytes()).hexdigest() == V2_SHA256
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(ROOT / "models/trustlaya-s-v2/model.safetensors")
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    print("Frozen v2 encoder loaded on", device, flush=True)
    train, tstat = tab_samples(tokenizer, "train"); dev, dstat = tab_samples(tokenizer, "dev")
    cache = ROOT / "benchmarks/external/predictions/v3_cache"
    cache.mkdir(parents=True, exist_ok=True)
    def cached(name, texts, token_features):
        path = cache / f"{name}.npy"
        if path.exists(): return np.load(path)
        features = encode(model, tokenizer, texts, device, token_features=token_features)
        np.save(path, features)
        return features
    ptrain = cached("tab_train_tokens", [r["text"] for r in train], True)
    pdev = cached("tab_dev_tokens", [r["text"] for r in dev], True)
    ty, tm, _ = pii_arrays(train, tokenizer); dy, dm, _ = pii_arrays(dev, tokenizer)
    pii_head, ph = train_pii(ptrain, ty, tm, pdev, dy, dm)
    save_file({k: v.detach().contiguous() for k, v in pii_head.state_dict().items()}, OUT / "pii_token_head.safetensors")
    del ptrain, pdev
    benign, bstat = benign_prompts()
    atrain = gandalf("train"); adev = gandalf("validation")
    xtexts = atrain + benign["train"]; dtexts = adev + benign["validation"]
    labels = np.array([1] * len(atrain) + [0] * len(benign["train"]), dtype=np.int8)
    dlabels = np.array([1] * len(adev) + [0] * len(benign["validation"]), dtype=np.int8)
    x = cached("attack_train_pooled", xtexts, False); dx = cached("attack_dev_pooled", dtexts, False)
    attack_head, ah = train_attack(x, labels, dx, dlabels)
    save_file({k: v.detach().contiguous() for k, v in attack_head.state_dict().items()}, OUT / "attack_head.safetensors")
    tokenizer.save_pretrained(OUT)
    manifest = {"version": "TrustLaya-S v3 external experimental", "v2_weight_sha256": hashlib.sha256((ROOT / "models/trustlaya-s-v2/model.safetensors").read_bytes()).hexdigest(),
                "pii_head_sha256": hashlib.sha256((OUT / "pii_token_head.safetensors").read_bytes()).hexdigest(),
                "attack_head_sha256": hashlib.sha256((OUT / "attack_head.safetensors").read_bytes()).hexdigest(),
                "seed": SEED, "encoder": "frozen v2", "pii_train": tstat, "pii_dev": dstat,
                "gandalf_train": len(atrain), "gandalf_dev": len(adev), "benign_selection": bstat,
                "pii_epochs": len(ph), "attack_epochs": len(ah), "max_length": MAXLEN,
                "pii_optimizer": "AdamW lr=0.002 batch=128 weighted cross entropy",
                "attack_optimizer": "AdamW lr=0.003 batch=128 balanced BCE",
                "pii_history": ph, "attack_history": ah,
                "source_versions": {"TAB": "558e09e26d6b36f5f78440074e6a233946d98bd9",
                                    "Gandalf": "04737b65e90a6794ec227012e4a255a7def6344b",
                                    "prompts.chat": "f78a1c5136fa080155d928e0d7e2b4a41ddef03e"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved", OUT, "— final test data never opened", flush=True)


if __name__ == "__main__": main()
