"""Merge two independent human reviews and a separate senior adjudication."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from trustlaya.v5_benchmark import ATTACK_VECTORS, BENIGN_TYPES, INTENTS, LOCATIONS, read_jsonl, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
FIELDS = ("sample_id", "annotator_id", "intent", "attack_vector", "benign_type",
          "language", "attack_location", "contains_pii", "contains_secret", "obfuscated")


def krippendorff_alpha_nominal(pairs: list[tuple[str, str]]) -> float | None:
    """Nominal alpha for complete two-rater units, including UNRESOLVED votes."""
    if not pairs:
        return None
    disagreements = sum(left != right for left, right in pairs)
    counts = Counter(label for pair in pairs for label in pair)
    n = 2 * len(pairs)
    expected = 1 - sum(count * (count - 1) for count in counts.values()) / (n * (n - 1))
    if expected == 0:
        return None  # No class diversity: chance-corrected reliability undefined.
    return 1 - (disagreements / len(pairs)) / expected


def read_reviews(path: Path) -> tuple[str, dict[str, dict]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS) <= set(reader.fieldnames or []):
            raise ValueError("annotation CSV missing hierarchical columns")
        rows = [row for row in reader if row["intent"].strip()]
    if not rows:
        raise ValueError("annotation file has no completed rows")
    annotators = {row["annotator_id"].strip() for row in rows}
    if len(annotators) != 1 or not next(iter(annotators)):
        raise ValueError("one nonempty annotator_id required per file")
    by_id = {}
    for row in rows:
        sample_id = row["sample_id"].strip()
        intent = row["intent"].strip()
        vector = row["attack_vector"].strip() or None
        benign_type = row["benign_type"].strip() or None
        location = row["attack_location"].strip() or None
        language = row["language"].strip().lower()
        if sample_id in by_id or not sample_id or intent not in INTENTS:
            raise ValueError(f"duplicate ID or invalid intent: {sample_id}")
        if (intent == "ATTACK") != (vector in ATTACK_VECTORS):
            raise ValueError(f"attack vector mismatch: {sample_id}")
        if (intent == "BENIGN_DUAL_USE") != (benign_type in BENIGN_TYPES):
            raise ValueError(f"benign subtype mismatch: {sample_id}")
        if vector and vector not in ATTACK_VECTORS or benign_type and benign_type not in BENIGN_TYPES:
            raise ValueError(f"invalid subtype: {sample_id}")
        if location and (location not in LOCATIONS or intent != "ATTACK"):
            raise ValueError(f"invalid location: {sample_id}")
        if language not in {"tr", "en", "mixed", "other", "undetermined"}:
            raise ValueError(f"invalid language: {sample_id}")
        flags = {}
        for field in ("contains_pii", "contains_secret", "obfuscated"):
            value = row[field].strip().lower()
            if value not in {"yes", "no"}:
                raise ValueError(f"{field} must be yes/no: {sample_id}")
            flags[field] = value == "yes"
        by_id[sample_id] = {"intent": intent, "attack_vector": vector,
                            "benign_type": benign_type, "language": language,
                            "attack_location": location, **flags}
    return next(iter(annotators)), by_id


def adjudicate(a_path: Path, b_path: Path, resolution_path: Path | None) -> dict:
    a_name, a = read_reviews(a_path)
    b_name, b = read_reviews(b_path)
    mapping_path = PRIVATE / "review_id_map.json"
    if not mapping_path.exists():
        raise FileNotFoundError("frozen blind review ID map missing")
    blind_to_source = json.loads(mapping_path.read_text())
    def map_ids(items: dict[str, dict]) -> dict[str, dict]:
        if not set(items) <= set(blind_to_source):
            raise ValueError("unknown blind review ID")
        return {blind_to_source[key]: value for key, value in items.items()}
    a, b = map_ids(a), map_ids(b)
    if a_name == b_name:
        raise ValueError("two distinct annotator IDs required")
    manifest = read_jsonl(ROOT / "benchmarks/v5/dataset_manifest.jsonl")
    ids = {row["sample_id"] for row in manifest}
    if not (set(a) <= ids and set(b) <= ids):
        raise ValueError("annotation contains unknown sample_id")
    senior_name, resolutions = None, {}
    if resolution_path:
        senior_name, resolutions = read_reviews(resolution_path)
        resolutions = map_ids(resolutions)
        if senior_name in {a_name, b_name}:
            raise ValueError("senior adjudicator must be independent")
        if not set(resolutions) <= (set(a) & set(b)):
            raise ValueError("adjudication requires two prior reviews")
    reviewed = Counter()
    pairs = []
    unresolved = []
    eligible_resolution = set()
    for row in manifest:
        sample_id = row["sample_id"]
        if sample_id not in a and sample_id not in b:
            continue
        if sample_id not in a or sample_id not in b:
            row["review_status"] = "ONE_REVIEW"
            reviewed["one_review"] += 1
            continue
        left, right = a[sample_id], b[sample_id]
        pairs.append((left["intent"], right["intent"]))
        if left == right and left["intent"] != "UNRESOLVED":
            chosen = left
            row["review_status"] = "AGREED"
            reviewed["agreed"] += 1
        else:
            eligible_resolution.add(sample_id)
            if left != right:
                reviewed["disagreement"] += 1
            if sample_id not in resolutions or resolutions[sample_id]["intent"] == "UNRESOLVED":
                row["review_status"] = "UNRESOLVED" if left["intent"] == right["intent"] else "DISAGREEMENT"
                unresolved.append(sample_id)
                continue
            chosen = resolutions[sample_id]
            row["review_status"] = "ADJUDICATED"
            reviewed["adjudicated"] += 1
        row["gold_intent"] = chosen["intent"]
        for field in ("attack_vector", "benign_type", "attack_location", "language",
                      "contains_pii", "contains_secret", "obfuscated"):
            row[field] = chosen[field]
    if set(resolutions) - eligible_resolution:
        raise ValueError("adjudication supplied for non-conflicting row")
    validate_manifest(manifest)
    (PRIVATE / "reviewed_manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    alpha = krippendorff_alpha_nominal(pairs)
    summary = {"annotator_ids": [a_name, b_name], "senior_adjudicator_id": senior_name,
               "reviewed": dict(reviewed), "both_reviewed": len(pairs),
               "intent_alpha_nominal": alpha,
               "disagreement_rate": reviewed["disagreement"] / len(pairs) if pairs else None,
               "unresolved_ids": unresolved, "gold_n": reviewed["agreed"] + reviewed["adjudicated"],
               "ready_for_freeze": len(pairs) >= 600 and alpha is not None and alpha >= 0.80 and not unresolved}
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
