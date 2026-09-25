"""V5 benchmark contract. Source labels are never promoted to human gold."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

LABELS = frozenset({
    "DIRECT_ATTACK", "PROMPT_INJECTION", "INDIRECT_ATTACK",
    "BENIGN_SECURITY_DISCUSSION", "QUOTED_ATTACK", "EDUCATIONAL_SECURITY",
    "NORMAL_REQUEST", "AMBIGUOUS",
})
ATTACK = frozenset({"DIRECT_ATTACK", "PROMPT_INJECTION", "INDIRECT_ATTACK"})
LOCATIONS = frozenset({
    "ATTACK_AT_BEGINNING", "ATTACK_IN_MIDDLE", "ATTACK_AT_END",
    "MULTIPLE_ATTACK_SEGMENTS", "DISTRIBUTED_CONTEXT",
})
BUCKETS = ((128, "0-128"), (256, "129-256"), (384, "257-384"),
           (512, "385-512"), (768, "513-768"), (1024, "769-1024"),
           (1536, "1025-1536"))


def bucket(length: int) -> str:
    if length < 0:
        raise ValueError("negative token length")
    return next((name for top, name in BUCKETS if length <= top), "1537+")


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_record(row: dict) -> None:
    required = ("sample_id", "source", "source_url", "original_dataset", "language",
                "provenance_type", "synthetic", "human_generated", "augmented",
                "source_label", "review_status", "split", "text_sha256",
                "normalized_sha256", "original_token_length", "model_visible_token_length",
                "truncated", "length_bucket")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if row["review_status"] not in {"UNREVIEWED", "ONE_REVIEW", "DISAGREEMENT", "ADJUDICATED", "AGREED"}:
        raise ValueError("invalid review status")
    if row["split"] not in {"TRAIN", "DEV", "HIDDEN_TEST_A", "HIDDEN_TEST_B", "CANDIDATE"}:
        raise ValueError("invalid split")
    if row.get("gold_label") is not None and (row["gold_label"] not in LABELS or row["review_status"] not in {"AGREED", "ADJUDICATED"}):
        raise ValueError("gold requires genuine completed human review")
    if row.get("attack_location") is not None and (row["attack_location"] not in LOCATIONS or row.get("gold_label") not in ATTACK):
        raise ValueError("attack location requires reviewed attack")
    if row["length_bucket"] != bucket(row["original_token_length"]):
        raise ValueError("incorrect length bucket")
    if row["model_visible_token_length"] > row["original_token_length"]:
        raise ValueError("visible length exceeds original")
    if row["truncated"] != (row["model_visible_token_length"] < row["original_token_length"]):
        raise ValueError("incorrect truncation flag")


def validate_manifest(rows: list[dict]) -> dict:
    ids = set()
    source_roles = defaultdict(set)
    hashes = defaultdict(set)
    counts = defaultdict(int)
    for row in rows:
        validate_record(row)
        if row["sample_id"] in ids:
            raise ValueError("duplicate sample_id")
        ids.add(row["sample_id"])
        source_roles[row["source"]].add(row["split"])
        hashes[row["normalized_sha256"]].add(row["split"])
        counts[row["split"]] += 1
    for source, roles in source_roles.items():
        if len(roles - {"CANDIDATE"}) > 1:
            raise ValueError(f"source appears in multiple assigned roles: {source}")
    contaminated = sum(len(roles) > 1 for roles in hashes.values())
    if contaminated:
        raise ValueError(f"normalized duplicate across roles: {contaminated}")
    return {"n": len(rows), "by_split": dict(counts), "human_reviewed": sum(r.get("gold_label") is not None for r in rows)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
