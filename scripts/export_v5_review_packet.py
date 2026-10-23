"""Make a source-blind local packet for two human annotators."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from trustlaya.v5_benchmark import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
SEED = 20260925


def main() -> None:
    metadata = {r["sample_id"]: r for r in read_jsonl(ROOT / "benchmarks/v5/dataset_manifest.jsonl")}
    queue = {r["sample_id"]: r for r in read_jsonl(PRIVATE / "annotation_queue.jsonl")}
    audit = json.loads((ROOT / "benchmarks/v5/contamination.json").read_text())
    excluded = {sample_id for item in audit["overlap"].values() for sample_id in item["candidate_ids"]}
    by_bucket = defaultdict(list)
    def length_group(length: int) -> str:
        return "0-94" if length <= 94 else "95-256" if length <= 256 else "257+"
    for sample_id, row in metadata.items():
        if row["source"] == "tensor_trust_game" and sample_id not in excluded:
            by_bucket[length_group(row["original_token_length"])].append(sample_id)
    rng = random.Random(SEED)
    # All 260 non-game candidates plus 340 game candidates. Equal quotas across
    # short/medium/long game inputs prevent a majority-wrapper sample.
    # Source and model scores are never provided to human annotators.
    selected = set()
    quotas = {"0-94": 114, "95-256": 113, "257+": 113}
    scores_path = PRIVATE / "candidate_disagreement.json"
    disagreement = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    for group, quota in quotas.items():
        ids = sorted(by_bucket[group])
        high = [sample_id for sample_id in ids if disagreement.get(sample_id, {}).get("disagreement")]
        low = [sample_id for sample_id in ids if sample_id not in high]
        high_quota = min(len(high), quota // 2)
        chosen = rng.sample(high, high_quota)
        chosen += rng.sample(low, min(len(low), quota - len(chosen)))
        if len(chosen) < quota:
            chosen += rng.sample(sorted(set(high) - set(chosen)), quota - len(chosen))
        selected.update(chosen)
    remainder = sorted({i for ids in by_bucket.values() for i in ids} - selected)
    selected.update(rng.sample(remainder, max(0, 340 - len(selected))))
    selected.update(sample_id for sample_id, row in metadata.items()
                    if row["source"] != "tensor_trust_game" and sample_id not in excluded)
    ids = sorted(selected)
    rng.shuffle(ids)
    review_ids = {sample_id: "r-" + hashlib.sha256(f"v5-packet-2:{sample_id}".encode()).hexdigest()[:20]
                  for sample_id in ids}
    if len(set(review_ids.values())) != len(ids):
        raise ValueError("review ID collision")
    packet = [{"sample_id": review_ids[sample_id], "text": queue[sample_id]["text"]} for sample_id in ids]
    (PRIVATE / "review_packet.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet))
    (PRIVATE / "review_id_map.json").write_text(json.dumps({review_ids[original]: original for original in ids}, indent=2) + "\n")
    for letter in ("a", "b"):
        with (PRIVATE / f"annotation_{letter}_template.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "annotator_id", "intent",
                                                        "attack_vector", "benign_type", "language",
                                                        "attack_location", "contains_pii", "contains_secret", "obfuscated"])
            writer.writeheader()
            writer.writerows({field: review_ids[sample_id] if field == "sample_id" else "" for field in writer.fieldnames}
                             for sample_id in ids)
    groups = Counter((metadata[i]["source"], length_group(metadata[i]["original_token_length"])) for i in ids)
    summary = {"packet_n": len(ids), "seed": SEED,
               "excluded_prior_near_overlap_n": len(excluded),
               "model_disagreement_stratified": bool(disagreement),
               "source_length_group_counts": {f"{source}:{group}": n for (source, group), n in sorted(groups.items())},
               "by_source": {source: sum(metadata[i]["source"] == source for i in ids)
                             for source in sorted({metadata[i]["source"] for i in ids})},
               "note": "Source and model scores are hidden from annotators; no gold labels exist yet."}
    (PRIVATE / "packet_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
