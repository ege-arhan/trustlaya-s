"""One-shot cross-source gate for released v2 and two isolated head candidates."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from train_advanced_pii import embeddings
from train_injection_candidate import DEST as BPI_CANDIDATE, ROOT, SOURCE, REVISION, key, load, measures
from select_injection_blend import DEST as BLEND_CANDIDATE
from train_injection_joint import DEST as JOINT_CANDIDATE
from trustlaya.dataset import read_rows
from trustlaya.model import BACKBONE, TrustLaya
from trustlaya.utils import device


POLYGUARD = "fevziegeyurtsevenler/PolyGuardBench"
POLYGUARD_REVISION = "45886b8e3ee89db4a08de767b304779995a2b26e"
AGENT_BENCH = "ppradyoth/AgentInjectionBench"
AGENT_REVISION = "ef230359966c4d87b1c925aa56b8bf6e3f6ffed3"


def polyguard_rows():
    path = Path(hf_hub_download(POLYGUARD, "data/all/train.jsonl", repo_type="dataset",
                                revision=POLYGUARD_REVISION))
    source_rows = [json.loads(line) for line in path.open()]
    training_keys = set()
    for split in ("train", "validation"):
        source = Path(hf_hub_download(SOURCE, f"{split}.jsonl", repo_type="dataset",
                                      revision=REVISION))
        training_keys.update(key(row["text"]) for row in
                             (json.loads(line) for line in source.open()))
    training_keys.update(key(row["text"]) for row in read_rows(ROOT / "data/splits/train.jsonl"))
    rows = []
    excluded = 0
    for row in source_rows:
        positive = row["axis"] == "prompt-injection" and row["label"] == "attack"
        negative = row["axis"] == "over-refusal" and row["label"] == "benign"
        if not (positive or negative):
            continue
        if key(row["text"]) in training_keys:
            excluded += 1
            continue
        rows.append({"text": row["text"], "label": int(positive),
                     "language": row["language"], "axis": row["axis"]})
    return rows, hashlib.sha256(path.read_bytes()).hexdigest(), excluded


def main():
    torch.set_num_threads(4)
    base = ROOT / "models/trustlaya-s-v2"
    directories = {"released_v2": base, "bpi_head": BPI_CANDIDATE}
    if (BLEND_CANDIDATE / "model.safetensors").exists():
        directories["development_blend"] = BLEND_CANDIDATE
    if (JOINT_CANDIDATE / "model.safetensors").exists():
        directories["joint_head"] = JOINT_CANDIDATE
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(base / "model.safetensors")
    model.to(device()).eval()
    tokenizer = AutoTokenizer.from_pretrained(base)
    bpi, bpi_hashes, bpi_excluded = load()
    polyguard, polyguard_hash, polyguard_excluded = polyguard_rows()
    agent_path = Path(hf_hub_download(AGENT_BENCH, "data/agent_injection_bench.jsonl",
                                      repo_type="dataset", revision=AGENT_REVISION))
    agent_rows = [json.loads(line) for line in agent_path.open()]
    agent = [{"text": "\n".join(str(turn.get("content") or "") for turn in row["conversation"]
                               if turn.get("role") == "tool_result"),
              "label": int(row["ground_truth"] == "unsafe")}
             for row in agent_rows]
    legacy = [{"text": row["text"], "label": row["labels"]["prompt_injection"]}
              for row in read_rows(ROOT / "data/splits/test.jsonl")]
    corpora = {"polyguard_cross_axis": polyguard, "bpi_test": bpi["test"],
               "legacy_synthetic": legacy, "agent_tool_results": agent}
    results = {}
    for corpus_name, rows in corpora.items():
        vectors = embeddings(model, tokenizer, rows)
        results[corpus_name] = {}
        for model_name, directory in directories.items():
            state = load_file(directory / "model.safetensors")
            temperature = json.loads((directory / "calibration.json").read_text())["prompt_injection"]
            threshold = (.5 if model_name == "released_v2" else
                         json.loads((directory / "decision_thresholds.json").read_text())["prompt_injection"])
            logits = vectors @ state["heads.risks.weight"][2].numpy() + float(
                state["heads.risks.bias"][2])
            summary = measures(rows, logits, temperature, threshold)
            if corpus_name == "polyguard_cross_axis":
                from scipy.special import expit
                pred = expit(logits / temperature) >= threshold
                summary["language"] = {lang: {
                    "attack_recall": float(np.mean([pred[i] for i, row in enumerate(rows)
                                                     if row["language"] == lang and row["label"]]))
                    if any(row["language"] == lang and row["label"] for row in rows) else None,
                    "benign_false_positive_rate": float(np.mean([pred[i] for i, row in enumerate(rows)
                                                                  if row["language"] == lang and not row["label"]]))
                    if any(row["language"] == lang and not row["label"] for row in rows) else None}
                    for lang in ("tr", "en")}
            results[corpus_name][model_name] = summary
    report = {"polyguard_source": POLYGUARD, "polyguard_revision": POLYGUARD_REVISION,
              "polyguard_license": "CC-BY-4.0", "polyguard_sha256": polyguard_hash,
              "polyguard_excluded_exact_training_overlap": polyguard_excluded,
              "polyguard_scope": "Only prompt-injection attack rows and over-refusal benign rows. They come from different axes, so combined F1 is a cross-axis diagnostic, not a matched injection benchmark. Report attack recall and benign FPR separately.",
              "bpi_source": SOURCE, "bpi_revision": REVISION,
              "bpi_test_sha256": bpi_hashes["test"],
              "bpi_excluded_test_overlap": bpi_excluded,
              "agent_source": AGENT_BENCH, "agent_revision": AGENT_REVISION,
              "agent_source_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
              "selection_note": "All candidate choices and thresholds were frozen on development splits before loading PolyGuardBench. BPI, legacy, and agent test metrics were already inspected in earlier experiments and are regression diagnostics, not fresh confirmation. A rejected blend with no saved weights is omitted.",
              "results": results}
    (ROOT / "reports/injection_candidate_holdout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {model: {k: value[k] for k in ("f1", "recall", "false_positive_rate")}
                             for model, value in group.items()}
                      for name, group in results.items()}, indent=2))


if __name__ == "__main__":
    main()
