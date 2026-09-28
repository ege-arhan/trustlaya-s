"""Frozen V2 diagnostic on Jev SILVER; agreement is not accuracy."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

from trustlaya.inference import Analyzer
from trustlaya.utils import normalize

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
INPUT = PRIVATE / "jev_silver_600_with_text.jsonl"
OUTPUT = PRIVATE / "v2_on_jev_silver_predictions.jsonl"
REPORT = ROOT / "benchmarks/v5/jev_v2_diagnostic.json"
V2 = ROOT / "models/trustlaya-s-v2"
ONNX = ROOT / "models/exported/v2/trustlaya_s.onnx"
V2_SHA = "99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c"
ONNX_SHA = "9d3b953709d990ef84b71c46f63dadec2076ca5b9f01cfd7c7d4ddc8e170526f"
THRESHOLD = 0.5  # frozen external baseline, not selected using Jev labels


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def compare(rows, label_key):
    """Agreement with a named reference; no ground-truth assertion."""
    rows = [r for r in rows if r[label_key] is not None]
    tp = sum(r[label_key] == 1 and r["v2_attack"] for r in rows)
    fp = sum(r[label_key] == 0 and r["v2_attack"] for r in rows)
    tn = sum(r[label_key] == 0 and not r["v2_attack"] for r in rows)
    fn = sum(r[label_key] == 1 and not r["v2_attack"] for r in rows)
    divide = lambda a, b: a / b if b else None
    return {
        "n": len(rows), "reference_positive": tp + fn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "agreement": divide(tp + tn, len(rows)),
        "precision_vs_reference": divide(tp, tp + fp),
        "recall_vs_reference": divide(tp, tp + fn),
        "f1_vs_reference": divide(2 * tp, 2 * tp + fp + fn),
        "v2_flag_rate_on_reference_negative": divide(fp, fp + tn),
    }


def percentile(values, pct):
    return float(np.percentile(values, pct)) if values else None


def overlap_diagnostic(texts):
    """Only against the recorded V2 project train split; no universal leakage claim."""
    normal = lambda value: re.sub(r"\s+", " ", value.casefold()).strip()
    train_path = ROOT / "data/splits/train.jsonl"
    train = [normal(json.loads(line)["text"]) for line in train_path.read_text().splitlines() if line]
    train_set = set(train)
    candidates = [normal(text) for text in texts]
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(4, 5), n_features=2 ** 18,
                                  alternate_sign=False, norm="l2")
    reference = vectorizer.transform(train)
    checked = vectorizer.transform(candidates)
    distance, _ = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1).fit(reference).kneighbors(checked)
    similarity = 1 - distance[:, 0]
    return {"scope": "data/splits/train.jsonl only", "train_n": len(train),
            "within_silver_normalized_duplicates": len(candidates) - len(set(candidates)),
            "normalized_exact_vs_v2_train": sum(text in train_set for text in candidates),
            "char_4_5_cosine_ge_0_85": int((similarity >= 0.85).sum()),
            "max_char_4_5_cosine_similarity": float(similarity.max())}


def ranking_diagnostic(rows, label_key):
    rows = [r for r in rows if r[label_key] is not None]
    labels = np.array([r[label_key] for r in rows], dtype=int)
    scores = np.array([r["v2_raw_score"] for r in rows], dtype=float)
    if len(set(labels)) != 2:
        return {"n": len(rows), "roc_auc_vs_reference": None,
                "average_precision_vs_reference": None}
    return {"n": len(rows), "roc_auc_vs_reference": float(roc_auc_score(labels, scores)),
            "average_precision_vs_reference": float(average_precision_score(labels, scores))}


def threshold_sweep(rows, label_key):
    rows = [r for r in rows if r[label_key] is not None]
    labels = np.array([r[label_key] for r in rows], dtype=int)
    scores = np.array([r["v2_raw_score"] for r in rows], dtype=float)
    result = []
    for i in range(101):
        threshold = i / 100
        pred = scores >= threshold
        tp = int(((labels == 1) & pred).sum())
        fp = int(((labels == 0) & pred).sum())
        fn = int(((labels == 1) & ~pred).sum())
        tn = int(((labels == 0) & ~pred).sum())
        result.append({"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                       "f1_vs_reference": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
    return result


def main():
    previous = json.loads((ROOT / "benchmarks/v5/jev_silver_manifest.json").read_text())
    if file_sha(INPUT) != previous["hashes"]["private_export_sha256"]:
        raise ValueError("Jev SILVER input changed")
    if file_sha(V2 / "model.safetensors") != V2_SHA or file_sha(ONNX) != ONNX_SHA:
        raise ValueError("frozen V2 artifact changed")
    source = [json.loads(line) for line in INPUT.read_text().splitlines() if line]
    if len(source) != 600 or len({r["sample_id"] for r in source}) != 600:
        raise ValueError("expected 600 unique SILVER rows")
    analyzer = Analyzer("onnx", model_dir=V2, onnx_path=ONNX)
    result = []
    for index, row in enumerate(source, 1):
        text = row["text"]
        if hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("source text mismatch")
        output = analyzer.analyze(text)
        token_length = len(analyzer.tokenizer.encode(normalize(text), add_special_tokens=False, verbose=False))
        intent = row["silver"]["intent"]
        if intent not in {"ATTACK", "BENIGN_DUAL_USE", "NORMAL", "UNRESOLVED"}:
            raise ValueError("invalid SILVER intent")
        raw = float(output["raw_scores"]["prompt_injection"])
        record = {
            "sample_id": row["sample_id"], "text_sha256": row["text_sha256"],
            "source": row["source"], "jev_intent": intent,
            "jev_binary": None if intent == "UNRESOLVED" else int(intent == "ATTACK"),
            "jev_max_probability": max(row["silver"]["intent_probabilities"].values()),
            "source_proxy_label": row.get("source_proxy_label"),
            "token_length_after_v2_normalize": token_length,
            "v2_input_truncated": token_length > 94,
            "v2_raw_score": raw,
            "v2_calibrated_score": float(output["calibrated_scores"]["prompt_injection"]),
            "v2_attack": int(raw >= THRESHOLD),
            "v2_policy_action": output["action"],
            "v2_evidence_types": sorted({e["type"] for e in output["evidence"]}),
            "v2_model_ms": output["timing_ms"]["model_inference"],
            "v2_total_ms": output["timing_ms"]["analysis_total"],
        }
        result.append(record)
        if index % 100 == 0:
            print(f"scored {index}/600", flush=True)
    with OUTPUT.open("w") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    OUTPUT.chmod(0o600)
    sources = {}
    for source_name in sorted({r["source"] for r in result}):
        subset = [r for r in result if r["source"] == source_name]
        sources[source_name] = {
            "n": len(subset),
            "jev_silver_attack": sum(r["jev_binary"] == 1 for r in subset),
            "v2_attack_at_frozen_threshold": sum(r["v2_attack"] for r in subset),
            "v2_policy_actions": dict(Counter(r["v2_policy_action"] for r in subset)),
            "agreement_vs_jev": compare(subset, "jev_binary"),
        }
    proxy = [r for r in result if r["source_proxy_label"] is not None]
    metrics = {
        "status": "DIAGNOSTIC_NOT_GOLD",
        "warning": "Jev SILVER and deepset train labels are not independent human GOLD. Agreement metrics are not model accuracy, calibration, or external generalization.",
        "model": "TrustLaya-S v2 frozen", "runtime": "ONNX CPU",
        "task": "prompt_injection raw head", "threshold": THRESHOLD,
        "input_n": len(result), "jev_unresolved_excluded": sum(r["jev_binary"] is None for r in result),
        "v2_attack_flagged": sum(r["v2_attack"] for r in result),
        "agreement_vs_jev": compare(result, "jev_binary"),
        "ranking_vs_jev": ranking_diagnostic(result, "jev_binary"),
        "threshold_sweep_vs_jev": threshold_sweep(result, "jev_binary"),
        "agreement_vs_deepset_train_source_proxy": compare(proxy, "source_proxy_label"),
        "ranking_vs_deepset_train_source_proxy": ranking_diagnostic(proxy, "source_proxy_label"),
        "threshold_sweep_vs_deepset_train_source_proxy": threshold_sweep(proxy, "source_proxy_label"),
        "sources": sources,
        "length": {
            "effective_content_tokens": 94,
            "over_94_n": sum(r["v2_input_truncated"] for r in result),
            "at_most_94_vs_jev": compare([r for r in result if not r["v2_input_truncated"]], "jev_binary"),
            "over_94_vs_jev": compare([r for r in result if r["v2_input_truncated"]], "jev_binary"),
        },
        "recorded_v2_train_overlap": overlap_diagnostic([r["text"] for r in source]),
        "latency_ms": {
            "v2_model_p50": percentile([r["v2_model_ms"] for r in result], 50),
            "v2_model_p95": percentile([r["v2_model_ms"] for r in result], 95),
            "v2_total_p50": percentile([r["v2_total_ms"] for r in result], 50),
            "v2_total_p95": percentile([r["v2_total_ms"] for r in result], 95),
        },
        "hashes": {
            "v2_model_sha256": V2_SHA, "v2_onnx_sha256": ONNX_SHA,
            "jev_private_input_sha256": previous["hashes"]["private_export_sha256"],
            "v2_predictions_sha256": file_sha(OUTPUT),
        },
    }
    REPORT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"n": len(result), "v2_flagged": metrics["v2_attack_flagged"],
                      "diagnostic": str(REPORT)}))


if __name__ == "__main__":
    main()
