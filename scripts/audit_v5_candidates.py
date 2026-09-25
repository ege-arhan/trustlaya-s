"""Audit candidate lengths and prior-data overlap without assigning labels."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from trustlaya.v5_benchmark import BUCKETS, normalized_hash, read_jsonl, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"


def previous_texts() -> dict[str, list[str]]:
    from run_external_real import jailbreak_samples
    clean_ids = {r["sample_id"] for r in json.loads((ROOT / "benchmarks/external/predictions/v4_frozen_inputs/v3_jailbreak_predictions.json").read_text())}
    return {
        "v2_train": [json.loads(line)["text"] for line in (ROOT / "data/splits/train.jsonl").open()],
        "v4_train": [r["text"] for r in json.loads((ROOT / "benchmarks/external/predictions/v4_train_text_private.json").read_text())],
        "v4_dev": [r["text"] for r in json.loads((ROOT / "benchmarks/external/predictions/v4_dev_text_private.json").read_text())],
        "jll_historical_clean": [r["text"] for r in jailbreak_samples()[0] if r["sample_id"] in clean_ids],
    }


def main() -> None:
    rows = read_jsonl(ROOT / "benchmarks/v5/dataset_manifest.jsonl")
    validate_manifest(rows)
    queue = read_jsonl(PRIVATE / "annotation_queue.jsonl")
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in queue]
    lengths = Counter(r["length_bucket"] for r in rows)
    sources = Counter(r["source"] for r in rows)
    trunc = sum(r["truncated"] for r in rows)
    token_lengths = np.array([r["original_token_length"] for r in rows])
    length_metrics = {
        "n": len(rows), "reviewed_gold_n": 0, "source_counts": dict(sources),
        "length_buckets": {name: lengths[name] for _, name in BUCKETS} | {"1537+": lengths["1537+"]},
        "truncated_at_current_94": trunc,
        "median_tokens": float(np.median(token_lengths)),
        "p90_tokens": float(np.percentile(token_lengths, 90)),
        "p95_tokens": float(np.percentile(token_lengths, 95)),
        "p99_tokens": float(np.percentile(token_lengths, 99)),
        "attack_rate": None, "benign_rate": None,
        "note": "Source labels are candidates, not human-reviewed gold labels."
    }
    (ROOT / "benchmarks/v5/length_metrics.json").write_text(json.dumps(length_metrics, indent=2) + "\n")
    texts = [r["text"] for r in queue]
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(4, 5), n_features=2**17,
                                  alternate_sign=False, norm="l2")
    candidate_vectors = vectorizer.transform(texts)
    previous = previous_texts()
    overlap = {}
    candidate_hashes = {r["normalized_sha256"] for r in rows}
    for source, items in previous.items():
        hashes = {normalized_hash(t) for t in items}
        matrix = vectorizer.transform(items)
        near = []
        for start in range(0, len(rows), 64):
            similarities = (candidate_vectors[start:start + 64] @ matrix.T).tocsr()
            for local in range(similarities.shape[0]):
                segment = similarities.getrow(local)
                if segment.nnz and segment.data.max() >= 0.85:
                    near.append(rows[start + local]["sample_id"])
        overlap[source] = {"previous_rows": len(items), "normalized_exact_unique": len(candidate_hashes & hashes),
                           "candidate_rows_char45_cosine_gte_0_85": len(near),
                           "candidate_ids": near[:100], "ids_truncated": len(near) > 100}
    within_exact = int(len(rows) - len({r["normalized_sha256"] for r in rows}))
    near_cross_source = 0
    pairs = (candidate_vectors @ candidate_vectors.T).tocsr()
    for index in range(len(rows)):
        part = pairs.getrow(index)
        near_cross_source += sum(index < other and score >= 0.85 and rows[index]["source"] != rows[other]["source"]
                                 for other, score in zip(part.indices, part.data))
    audit = {"candidate_n": len(rows), "status": "candidate-only; hidden tests not yet acquired or inspected",
             "method": "normalized exact plus hashed char 4-5gram cosine >=0.85",
             "coverage": "v2 train, v4 train/dev, historical JLL clean; no hidden sources acquired",
             "within_candidate_normalized_exact_extra_rows": within_exact,
             "within_candidate_cross_source_near_pairs_gte_0_85": int(near_cross_source),
             "overlap": overlap}
    (ROOT / "benchmarks/v5/contamination.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"length": length_metrics, "overlap": {k: {x: y for x, y in v.items() if x != "candidate_ids"} for k, v in overlap.items()}}, indent=2))


if __name__ == "__main__":
    main()
