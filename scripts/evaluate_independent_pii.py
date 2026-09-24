"""Untouched CC-BY Turkish PII test; aggregates only, no row redistribution."""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer
from scipy.special import expit
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from train_advanced_pii import embeddings, evaluate
from trustlaya.evidence import extract, PII_TYPES
from trustlaya.model import TrustLaya, BACKBONE
from trustlaya.utils import device, seed_all

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "BTX24/turkish-privacy-pii-ner"
SOURCE_REVISION = "45a8e18edb57224b14368c910f701e064dbb1326"
PII_CATEGORIES = ("account_number", "private_phone", "tckn", "private_email",
                  "vkn", "private_address", "iban", "private_person")
NEGATIVE_CATEGORIES = ("private_date", "secret")


def main():
    seed_all(42)
    torch.set_num_threads(4)
    dataset = load_dataset(SOURCE, split="test", revision=SOURCE_REVISION)
    grouped = defaultdict(list)
    for row in dataset:
        categories = {item["category"] for item in row["label"]}
        if len(categories) == 1:
            category = next(iter(categories))
            if category in PII_CATEGORIES + NEGATIVE_CATEGORIES:
                grouped[category].append(row)
    rng = random.Random(42)
    rows = []
    for category in PII_CATEGORIES:
        rows.extend({"text": row["text"], "label": 1, "category": category}
                    for row in rng.sample(grouped[category], 125))
    for category in NEGATIVE_CATEGORIES:
        rows.extend({"text": row["text"], "label": 0, "category": category}
                    for row in rng.sample(grouped[category], 500))
    rng.shuffle(rows)
    labels = np.array([row["label"] for row in rows])
    rule_predictions = np.array([any(item["type"] in PII_TYPES for item in extract(row["text"]))
                                 for row in rows], dtype=bool)
    tn, fp, fn, tp = confusion_matrix(labels, rule_predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, rule_predictions, average="binary", zero_division=0)
    rule_baseline = {"n": len(rows), "positive": int(labels.sum()),
                     "precision": float(precision), "recall": float(recall), "f1": float(f1),
                     "false_positive_rate": float(fp / max(1, fp + tn)),
                     "false_negative_rate": float(fn / max(1, fn + tp)),
                     "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v1")
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(ROOT / "models/trustlaya-s-v1/model.safetensors")
    model.to(device()).eval()
    vectors = embeddings(model, tokenizer, rows)
    results = {}
    for name in ("v1", "v2"):
        folder = ROOT / f"models/trustlaya-s-{name}"
        weights = load_file(str(folder / "model.safetensors"))
        logits = vectors @ weights["heads.risks.weight"][0].numpy() + float(
            weights["heads.risks.bias"][0])
        temperature = json.loads((folder / "calibration.json").read_text())["pii"]
        threshold = .5 if name == "v1" else json.loads(
            (folder / "decision_thresholds.json").read_text())["pii"]
        results[name] = {"at_0_5": evaluate(logits, rows, temperature),
                         "dev_tuned": evaluate(logits, rows, temperature, threshold)}
        probabilities = expit(logits / temperature)
        rules = [any(item["type"] in PII_TYPES for item in extract(row["text"]))
                 for row in rows]
        predicted = [(score >= threshold) or rule for score, rule in zip(probabilities, rules)]
        results[name]["by_category"] = {
            category: {"n": sum(row["category"] == category for row in rows),
                       "predicted_positive": int(sum(out for row, out in zip(rows, predicted)
                                                     if row["category"] == category)),
                       "rule_positive": int(sum(out for row, out in zip(rows, rules)
                                                if row["category"] == category))}
            for category in PII_CATEGORIES + NEGATIVE_CATEGORIES
        }
    report = {
        "source": SOURCE, "source_revision": SOURCE_REVISION, "license": "CC-BY-4.0",
        "scope": "Test split, seed-42 stratified 125 per eight PII categories and 500 each of date and secret task-specific negatives. Entire source is synthetic; no raw examples redistributed. Date and secret are not globally benign.",
        "n": len(rows), "positive": 1000, "negative": 1000,
        "results": {"rule_only": rule_baseline, **results},
        "selection_note": "This dataset was not used for training, calibration or threshold selection. After inspection, it is no longer a fresh future holdout.",
    }
    (ROOT / "reports/independent_pii_v2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
