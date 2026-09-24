"""One-shot cross-source diagnostic on an untouched grouped test split."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from trustlaya.inference import Analyzer
from trustlaya.dataset import read_rows
from train_injection_candidate import key, load as load_bpi
from train_injection_agentic import agentic

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "neuralchemy/Prompt-injection-dataset"
REVISION = "7d70432dfcf47a821612cbf9d34e9d9e3ad20e75"
FILE = "core/test-00000-of-00001.parquet"


def main():
    path = Path(hf_hub_download(SOURCE, FILE, repo_type="dataset", revision=REVISION))
    rows = pq.read_table(path).to_pylist()
    assert all(not row["augmented"] for row in rows)
    bpi, _, _ = load_bpi()
    training_text = [row["text"] for split in ("train", "validation")
                     for row in bpi[split]]
    for split in ("train", "validation"):
        source_rows, _ = agentic(split)
        training_text.extend(row["text"] for row in source_rows)
    for split in ("train", "val"):
        training_text.extend(row["text"] for row in read_rows(ROOT / f"data/splits/{split}.jsonl"))
    training_keys = {key(text) for text in training_text}
    excluded = [row for row in rows if key(row["text"]) in training_keys]
    rows = [row for row in rows if key(row["text"]) not in training_keys]
    labels = [int(row["label"]) for row in rows]
    assert set(labels) == {0, 1}
    report = {
        "source": SOURCE, "revision": REVISION, "file": FILE,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "license": "Apache-2.0; source-level terms may vary",
        "scope": "Broad prompt injection and jailbreak versus benign, not strict tool-return injection",
        "rows": len(rows), "label_counts": dict(Counter(labels)),
        "excluded_exact_or_normalized_training_overlap": len(excluded),
        "excluded_label_counts": dict(Counter(int(row["label"]) for row in excluded)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "models": {},
    }
    for name, folder, onnx, threshold in (
        ("released_v2", "models/trustlaya-s-v2", "models/exported/v2/trustlaya_s.onnx", .5),
        ("agentic_candidate", "models/candidates/injection_agentic",
         "models/exported/injection_agentic/trustlaya_s.onnx", .65),
    ):
        analyzer = Analyzer("onnx", model_dir=ROOT / folder, onnx_path=ROOT / onnx)
        predictions = [int(analyzer.analyze(row["text"])["calibrated_scores"]["prompt_injection"] >= threshold)
                       for row in rows]
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        report["models"][name] = {
            "threshold": threshold, "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
            "false_positive_rate": fp / (fp + tn),
            "false_negative_rate": fn / (fn + tp),
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
            "by_source": {},
        }
        for source in sorted(set(row["source"] for row in rows)):
            indices = [i for i, row in enumerate(rows) if row["source"] == source]
            source_truth = [labels[i] for i in indices]
            source_pred = [predictions[i] for i in indices]
            source_tn, source_fp, source_fn, source_tp = confusion_matrix(
                source_truth, source_pred, labels=[0, 1]).ravel()
            report["models"][name]["by_source"][source] = {
                "n": len(indices), "positive": sum(source_truth),
                "f1": f1_score(source_truth, source_pred, zero_division=0),
                "false_positive_rate": (source_fp / (source_fp + source_tn)
                                        if source_fp + source_tn else None),
                "false_negative_rate": (source_fn / (source_fn + source_tp)
                                        if source_fn + source_tp else None),
                "confusion_matrix_tn_fp_fn_tp": [int(source_tn), int(source_fp),
                                                 int(source_fn), int(source_tp)],
            }
        del analyzer
    output = ROOT / "reports/neuralchemy_cross_source.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
