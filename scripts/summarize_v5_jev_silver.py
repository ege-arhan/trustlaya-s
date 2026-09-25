"""Audit Jev-only SILVER output without treating it as gold truth."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from transformers import AutoTokenizer

try:
    from scripts.run_v5_jev_silver import ROOT, QUESTIONS, original_permitted_rows, replacement_rows, sha256
except ModuleNotFoundError:  # direct `python scripts/...` entry point
    from run_v5_jev_silver import ROOT, QUESTIONS, original_permitted_rows, replacement_rows, sha256


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    return values[round((len(values) - 1) * percent / 100)]


def summarize(path: Path):
    original, original_sha = original_permitted_rows()
    replacement, replacement_sha = replacement_rows(original)
    expected = {r["sample_id"]: r for r in original + replacement}
    data = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(data) != 600 or len({r["sample_id"] for r in data}) != 600:
        raise ValueError("expected exactly 600 unique predictions")
    if {r["sample_id"] for r in data} != set(expected):
        raise ValueError("prediction IDs differ from selected packet")
    for row in data:
        if row["text_sha256"] != expected[row["sample_id"]]["text_sha256"]:
            raise ValueError("prediction text hash differs from selected packet")
        if row["silver"]["annotation_source"] != "JEV_SILVER":
            raise ValueError("non-Jev annotation")
    by_source = {}
    for source in sorted({r["source"] for r in data}):
        subset = [r for r in data if r["source"] == source]
        by_source[source] = {
            "n": len(subset),
            "silver_intent": dict(Counter(r["silver"]["intent"] for r in subset)),
            "quality_flags": dict(Counter(flag for r in subset for flag in r["silver"]["quality_flags"])),
        }
    proxy = [r for r in data if "source_proxy_label" in r]
    agreement = sum(
        (r["silver"]["intent"] == "ATTACK") if r["source_proxy_label"] == 1
        else (r["silver"]["intent"] in {"NORMAL", "BENIGN_DUAL_USE"})
        for r in proxy
    )
    proxy_cross_tab = {
        str(label): dict(Counter(r["silver"]["intent"] for r in proxy if r["source_proxy_label"] == label))
        for label in (0, 1)
    }
    latency = [r["latency_ms"] for r in data]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v2", local_files_only=True)
    visible_content_limit = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=False)
    lengths = {sid: len(tokenizer.encode(row["text"], add_special_tokens=False, verbose=False))
               for sid, row in expected.items()}
    buckets = [(0, 128), (129, 256), (257, 384), (385, 512), (513, 768),
               (769, 1024), (1025, 1536), (1537, None)]
    bucket_counts = {}
    for lower, upper in buckets:
        ids = {sid for sid, n in lengths.items() if n >= lower and (upper is None or n <= upper)}
        bucket_counts[f"{lower}-{upper if upper is not None else 'plus'}"] = {
            "n": len(ids),
            "silver_attack": sum(r["silver"]["intent"] == "ATTACK" for r in data if r["sample_id"] in ids),
        }
    total_input = sum((r.get("usage") or {}).get("inputTokens", 0) for r in data)
    total_output = sum((r.get("usage") or {}).get("outputTokens", 0) for r in data)
    costs = [r.get("gateway_cost") for r in data]
    return {
        "status": "JEV_SILVER_NOT_GOLD", "n": len(data),
        "human_labels": 0, "raw_text_in_public_manifest": False,
        "model_alias": "typesafe-ai/jev", "source_counts": by_source,
        "overall_silver_intent": dict(Counter(r["silver"]["intent"] for r in data)),
        "deepset_source_proxy_agreement": {"agree": agreement, "n": len(proxy),
            "warning": "Source labels are not an independent human-gold evaluation of Jev."},
        "deepset_source_proxy_cross_tab": proxy_cross_tab,
        "latency_ms": {"p50": percentile(latency, 50), "p95": percentile(latency, 95),
                       "p99": percentile(latency, 99)},
        "length": {"v2_max_sequence_length": tokenizer.model_max_length,
                   "v2_visible_content_tokens": visible_content_limit,
                   "median_tokens": percentile(list(lengths.values()), 50),
                   "p95_tokens": percentile(list(lengths.values()), 95),
                   "truncated_by_v2_head_window": sum(n > visible_content_limit for n in lengths.values()),
                   "buckets": bucket_counts},
        "usage": {"input_tokens": total_input, "output_tokens": total_output},
        "gateway_reported_cost": str(sum(Decimal(str(c)) for c in costs if c is not None)) if any(c is not None for c in costs) else None,
        "hashes": {"original_packet_sha256": original_sha,
                   "replacement_packet_sha256": replacement_sha,
                   "predictions_sha256": sha256(path.read_bytes()),
                   "jev_questions_sha256": sha256(json.dumps(QUESTIONS, sort_keys=True).encode()),
                   "v2_tokenizer_sha256": sha256((ROOT / "models/trustlaya-s-v2/tokenizer.json").read_bytes())},
        "deepset_revision": "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        "selection_seed": 5025,
        "test_split_used": False,
    }


def export_private(predictions: Path, destination: Path):
    """Join licensed candidate text to Jev labels for local research use only."""
    original, _ = original_permitted_rows()
    replacement, _ = replacement_rows(original)
    by_id = {r["sample_id"]: r for r in original + replacement}
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
    if len(rows) != 600 or len({r["sample_id"] for r in rows}) != 600:
        raise ValueError("expected 600 unique silver labels")
    if set(by_id) != {r["sample_id"] for r in rows}:
        raise ValueError("selection mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        for row in rows:
            source = by_id[row["sample_id"]]
            if source["text_sha256"] != row["text_sha256"]:
                raise ValueError("text hash mismatch")
            joined = {"sample_id": row["sample_id"], "text": source["text"],
                      "source": row["source"], "source_url": row["source_url"],
                      "license": row["license"], "text_sha256": row["text_sha256"],
                      "silver": row["silver"]}
            if "source_proxy_label" in row:
                joined["source_proxy_label"] = row["source_proxy_label"]
            stream.write(json.dumps(joined, ensure_ascii=False) + "\n")
    destination.chmod(0o600)
    return sha256(destination.read_bytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT / "benchmarks/v5/private/jev_silver_600.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/v5/jev_silver_manifest.json")
    parser.add_argument("--private-export", type=Path, default=ROOT / "benchmarks/v5/private/jev_silver_600_with_text.jsonl")
    args = parser.parse_args()
    result = summarize(args.predictions)
    result["hashes"]["private_export_sha256"] = export_private(args.predictions, args.private_export)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n": result["n"], "intent": result["overall_silver_intent"],
                      "manifest": str(args.output)}))


if __name__ == "__main__":
    main()
