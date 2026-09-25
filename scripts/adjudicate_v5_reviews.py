"""Import two independent human annotation files; never infer missing labels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from trustlaya.v5_benchmark import ATTACK, LABELS, LOCATIONS, read_jsonl, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"


def read_reviews(path: Path) -> tuple[str, dict[str, dict]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"sample_id", "annotator_id", "label", "language", "attack_location"} <= set(reader.fieldnames or []):
            raise ValueError("annotation CSV missing required columns")
        rows = list(reader)
    annotators = {r["annotator_id"].strip() for r in rows}
    if len(annotators) != 1 or not next(iter(annotators)):
        raise ValueError("one nonempty annotator_id required per file")
    by_id = {}
    for row in rows:
        sample_id, label = row["sample_id"].strip(), row["label"].strip()
        location = row["attack_location"].strip()
        language = row["language"].strip().lower()
        if sample_id in by_id or label not in LABELS:
            raise ValueError(f"duplicate or invalid label: {sample_id}")
        if location and (location not in LOCATIONS or label not in ATTACK):
            raise ValueError(f"invalid attack location: {sample_id}")
        if language not in {"tr", "en", "mixed", "other", "undetermined"}:
            raise ValueError(f"invalid language: {sample_id}")
        by_id[sample_id] = {"label": label, "language": language, "attack_location": location or None}
    return next(iter(annotators)), by_id


def adjudicate(a_path: Path, b_path: Path, resolution_path: Path | None) -> dict:
    a_name, a = read_reviews(a_path)
    b_name, b = read_reviews(b_path)
    if a_name == b_name:
        raise ValueError("two distinct annotator IDs required")
    manifest = read_jsonl(ROOT / "benchmarks/v5/dataset_manifest.jsonl")
    ids = {r["sample_id"] for r in manifest}
    if not (set(a) <= ids and set(b) <= ids):
        raise ValueError("annotation contains unknown sample_id")
    resolutions = {}
    if resolution_path:
        _, resolutions = read_reviews(resolution_path)
    conflicts = []
    reviewed = Counter()
    for row in manifest:
        sample_id = row["sample_id"]
        if sample_id not in a and sample_id not in b:
            continue
        if sample_id not in a or sample_id not in b:
            row["review_status"] = "ONE_REVIEW"
            reviewed["one_review"] += 1
            continue
        if a[sample_id] == b[sample_id]:
            row["review_status"] = "AGREED"
            chosen = a[sample_id]
            reviewed["agreed"] += 1
        else:
            reviewed["disagreement"] += 1
            conflicts.append(sample_id)
            if sample_id not in resolutions:
                row["review_status"] = "DISAGREEMENT"
                continue
            row["review_status"] = "ADJUDICATED"
            chosen = resolutions[sample_id]
            reviewed["adjudicated"] += 1
        row["gold_label"] = chosen["label"]
        row["language"] = chosen["language"]
        row["attack_location"] = chosen["attack_location"]
    validate_manifest(manifest)
    output = PRIVATE / "reviewed_manifest.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    summary = {"annotator_ids": [a_name, b_name], "reviewed": dict(reviewed),
               "both_reviewed": reviewed["agreed"] + reviewed["disagreement"],
               "disagreement_rate": reviewed["disagreement"] / max(1, reviewed["agreed"] + reviewed["disagreement"]),
               "unresolved_conflict_ids": [x for x in conflicts if x not in resolutions],
               "gold_n": reviewed["agreed"] + reviewed["adjudicated"]}
    (PRIVATE / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", required=True, type=Path)
    parser.add_argument("--annotator-b", required=True, type=Path)
    parser.add_argument("--adjudication", type=Path)
    args = parser.parse_args()
    print(json.dumps(adjudicate(args.annotator_a, args.annotator_b, args.adjudication), indent=2))


if __name__ == "__main__":
    main()
