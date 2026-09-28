"""Compare an isolated injection candidate on cross-source and legacy cases."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from train_advanced_pii import embeddings
from train_injection_candidate import key, measures, DEST, ROOT, SOURCE, REVISION
from trustlaya.dataset import read_rows
from trustlaya.model import BACKBONE, TrustLaya
from trustlaya.utils import device

FRESH_SOURCE = "zachz/prompt-injection-benchmark"
FRESH_REVISION = "dc0072ad6af8d1be049cdb60959e3ff51df4bcae"


def external_rows():
    path = Path(hf_hub_download(FRESH_SOURCE, "data.csv", repo_type="dataset",
                                revision=FRESH_REVISION))
    rows = list(csv.DictReader(path.open()))
    training_keys = set()
    for split in ("train", "validation"):
        source = Path(hf_hub_download(SOURCE, f"{split}.jsonl", repo_type="dataset",
                                      revision=REVISION))
        training_keys.update(key(row["text"]) for row in
                             (json.loads(line) for line in source.open()))
    clean = [{"text": row["text"], "label": int(row["label"] == "injection"),
              "category": row.get("category")}
             for row in rows if row["label"] in ("benign", "injection")
             and key(row["text"]) not in training_keys]
    return clean, hashlib.sha256(path.read_bytes()).hexdigest(), len(rows) - len(clean)


def head_metrics(vectors, rows, weight, bias, temperature, threshold):
    logits = vectors @ weight + bias
    return measures(rows, logits, temperature, threshold)


def main():
    base = ROOT / "models/trustlaya-s-v2"
    if not (DEST / "model.safetensors").exists():
        raise FileNotFoundError("Train the isolated candidate first")
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(base / "model.safetensors")
    model.to(device()).eval()
    tokenizer = AutoTokenizer.from_pretrained(base)
    base_state = load_file(base / "model.safetensors")
    candidate_state = load_file(DEST / "model.safetensors")
    temperatures = [json.loads((directory / "calibration.json").read_text())["prompt_injection"]
                    for directory in (base, DEST)]
    threshold = json.loads((DEST / "decision_thresholds.json").read_text())["prompt_injection"]
    fresh, source_hash, overlap = external_rows()
    legacy = [{"text": row["text"], "label": row["labels"]["prompt_injection"]}
              for row in read_rows(ROOT / "data/splits/test.jsonl")]
    output = {}
    for name, rows in (("fresh_zachz", fresh), ("legacy_synthetic", legacy)):
        vectors = embeddings(model, tokenizer, rows)
        settings = (("v2_at_0_5", base_state, temperatures[0], .5),
                    ("candidate_at_0_5", candidate_state, temperatures[1], .5),
                    ("candidate_dev_threshold", candidate_state, temperatures[1], threshold))
        output[name] = {label: head_metrics(vectors, rows,
                                           state["heads.risks.weight"][2].numpy(),
                                           float(state["heads.risks.bias"][2]), temp, cut)
                        for label, state, temp, cut in settings}
    report = {"fresh_source": FRESH_SOURCE, "fresh_source_revision": FRESH_REVISION,
              "fresh_license": "MIT", "fresh_sha256": source_hash,
              "fresh_excluded_exact_training_overlap": overlap,
              "selection_note": "Fresh source and legacy synthetic test are diagnostics after BPI validation-only candidate selection; no threshold or C was chosen here. Fresh source has short, English-heavy authored/template examples; it is not agent tool-output traffic.",
              "results": output}
    (ROOT / "reports/injection_candidate_cross_source.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {label: {k: value[k] for k in ("f1", "recall", "false_positive_rate")}
                              for label, value in group.items()}
                      for name, group in output.items()}, indent=2))


if __name__ == "__main__":
    main()
