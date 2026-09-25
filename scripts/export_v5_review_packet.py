"""Make a source-blind local packet for two human annotators."""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
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
    for sample_id, row in metadata.items():
        if row["source"] == "tensor_trust_game" and sample_id not in excluded:
            by_bucket[row["length_bucket"]].append(sample_id)
    rng = random.Random(SEED)
    # 200 game attempts, with long examples intentionally represented. These
    # are source candidates only; no attack label is inferred from selection.
    selected = set()
    for length_bucket in sorted(by_bucket):
        ids = sorted(by_bucket[length_bucket])
        selected.update(rng.sample(ids, min(25, len(ids))))
    remainder = sorted({i for ids in by_bucket.values() for i in ids} - selected)
    selected.update(rng.sample(remainder, max(0, 200 - len(selected))))
    selected.update(sample_id for sample_id, row in metadata.items()
                    if row["source"] != "tensor_trust_game" and sample_id not in excluded)
    ids = sorted(selected)
    rng.shuffle(ids)
    packet = [{"sample_id": sample_id, "text": queue[sample_id]["text"]} for sample_id in ids]
    (PRIVATE / "review_packet.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet))
    for letter in ("a", "b"):
        with (PRIVATE / f"annotation_{letter}_template.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sample_id", "annotator_id", "label", "language", "attack_location"])
            writer.writeheader()
            writer.writerows({"sample_id": sample_id, "annotator_id": "", "label": "", "language": "", "attack_location": ""}
                             for sample_id in ids)
    summary = {"packet_n": len(ids), "seed": SEED,
               "excluded_prior_near_overlap_n": len(excluded),
               "by_source": {source: sum(metadata[i]["source"] == source for i in ids)
                             for source in sorted({metadata[i]["source"] for i in ids})},
               "note": "Source and model scores are hidden from annotators; no gold labels exist yet."}
    (PRIVATE / "packet_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
