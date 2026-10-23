"""One-shot agentic and broad attack diagnostics for frozen injection heads."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from train_advanced_pii import embeddings
from train_injection_candidate import DEST as BPI_CANDIDATE, ROOT, SOURCE as BPI, REVISION as BPI_REVISION, key, measures
from train_injection_joint import DEST as JOINT_CANDIDATE
from train_injection_agentic import DEST as AGENT_CANDIDATE, SOURCE as AGENT, REVISION as AGENT_REVISION
from trustlaya.dataset import read_rows
from trustlaya.model import BACKBONE, TrustLaya
from trustlaya.utils import device


PROMPTWALL = "cyberec/promptwall-injection-dataset"
PROMPTWALL_REVISION = "ff1840ff2e36a8037fd5d851c88d04d3d08b028f"


def training_keys():
    keys = {key(row["text"]) for row in read_rows(ROOT / "data/splits/train.jsonl")}
    for split in ("train", "validation"):
        path = Path(hf_hub_download(BPI, f"{split}.jsonl", repo_type="dataset",
                                    revision=BPI_REVISION))
        keys.update(key(row["text"]) for row in
                    (json.loads(line) for line in path.open()))
    path = Path(hf_hub_download(AGENT, "data/train.parquet", repo_type="dataset",
                                revision=AGENT_REVISION))
    keys.update(key(row["text"]) for row in parquet.read_table(path).to_pylist())
    return keys


def sources():
    blocked = training_keys()
    path = Path(hf_hub_download(AGENT, "data/test.parquet", repo_type="dataset",
                                revision=AGENT_REVISION))
    original_agent = parquet.read_table(path).to_pylist()
    agent = [{"text": row["text"], "label": int(row["label"]),
              "family": row["attack_family"]}
             for row in original_agent if key(row["text"]) not in blocked]
    agent_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    wall, wall_hashes, wall_excluded = [], {}, {}
    for file, label in (("attacks.jsonl", 1), ("safe.jsonl", 0)):
        path = Path(hf_hub_download(PROMPTWALL, file, repo_type="dataset",
                                    revision=PROMPTWALL_REVISION))
        raw = [json.loads(line) for line in path.open()]
        wall_hashes[file] = hashlib.sha256(path.read_bytes()).hexdigest()
        wall_excluded[file] = sum(key(row["prompt"]) in blocked for row in raw)
        wall.extend({"text": row["prompt"], "label": label,
                     "family": row.get("attack_type", "safe")}
                    for row in raw if key(row["prompt"]) not in blocked)
    return {"agentic_paired_test": agent, "promptwall_broad_test": wall}, {
        "agentic_test_sha256": agent_hash,
        "agentic_exact_training_overlap_excluded": len(original_agent) - len(agent),
        "promptwall_sha256": wall_hashes,
        "promptwall_exact_training_overlap_excluded": wall_excluded}


def main():
    torch.set_num_threads(4)
    base = ROOT / "models/trustlaya-s-v2"
    directories = {"released_v2": base, "bpi_head": BPI_CANDIDATE,
                   "joint_head": JOINT_CANDIDATE, "agentic_head": AGENT_CANDIDATE}
    for path in directories.values():
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(path / "model.safetensors")
    rows_by_source, provenance = sources()
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(base / "model.safetensors")
    model.to(device()).eval()
    tokenizer = AutoTokenizer.from_pretrained(base)
    results = {}
    for corpus, rows in rows_by_source.items():
        vectors = embeddings(model, tokenizer, rows)
        results[corpus] = {}
        for name, path in directories.items():
            state = load_file(path / "model.safetensors")
            temperature = json.loads((path / "calibration.json").read_text())["prompt_injection"]
            threshold = (.5 if name == "released_v2" else
                         json.loads((path / "decision_thresholds.json").read_text())["prompt_injection"])
            logits = vectors @ state["heads.risks.weight"][2].numpy() + float(
                state["heads.risks.bias"][2])
            summary = measures(rows, logits, temperature, threshold)
            if corpus == "agentic_paired_test":
                from scipy.special import expit
                predictions = expit(logits / temperature) >= threshold
                families = defaultdict(list)
                for index, row in enumerate(rows):
                    families[row["family"]].append(index)
                summary["family_recall"] = {
                    family: float(np.mean([predictions[i] for i in indices if rows[i]["label"]]))
                    for family, indices in families.items()
                    if any(rows[i]["label"] for i in indices)}
                summary["benign_fpr"] = float(np.mean([predictions[i]
                                                        for i, row in enumerate(rows) if not row["label"]]))
            results[corpus][name] = summary
    report = {"agentic_source": AGENT, "agentic_revision": AGENT_REVISION,
              "agentic_license": "CC-BY-4.0", "agentic_attribution": "Enes Deniz",
              "promptwall_source": PROMPTWALL, "promptwall_revision": PROMPTWALL_REVISION,
              "promptwall_license": "MIT", **provenance,
              "scope": "Agentic5K is paired synthetic attack/benign data with scenario-isolated source splits. PromptWall has broad attack/jailbreak labels and generic safe controls, not matched tool-return benign data. Neither measures agent execution. All variants and thresholds were frozen before loading these test files.",
              "selection_note": "No parameter, threshold or model selection uses these tests. Once inspected, they are not fresh for further tuning.",
              "results": results}
    (ROOT / "reports/injection_agentic_holdout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({corpus: {name: {metric: value[metric]
                                        for metric in ("f1", "recall", "false_positive_rate")}
                               for name, value in group.items()}
                      for corpus, group in results.items()}, indent=2))


if __name__ == "__main__":
    main()
