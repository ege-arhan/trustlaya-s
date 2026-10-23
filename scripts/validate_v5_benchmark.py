"""Gate: a candidate pool cannot masquerade as a completed external benchmark."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from trustlaya.v5_benchmark import read_jsonl, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def readiness(rows: list[dict]) -> dict:
    summary = validate_manifest(rows)
    by_role = defaultdict(list)
    for row in rows:
        by_role[row["split"]].append(row)
    problems = []
    for role in ("TRAIN", "DEV", "HIDDEN_TEST_A", "HIDDEN_TEST_B"):
        part = by_role[role]
        if not part:
            problems.append(f"{role}_EMPTY")
            continue
        reviewed = [r for r in part if r.get("gold_intent") in {"ATTACK", "BENIGN_DUAL_USE", "NORMAL"}]
        if len(reviewed) < len(part):
            problems.append(f"{role}_UNREVIEWED_OR_AMBIGUOUS")
        classes = Counter(r["gold_intent"] == "ATTACK" for r in reviewed)
        if min(classes[True], classes[False]) < 20:
            problems.append(f"{role}_INSUFFICIENT_CLASS_COVERAGE")
    summary_path = ROOT / "benchmarks/v5/private/review_summary.json"
    if not summary_path.exists():
        problems.append("TWO_HUMAN_REVIEWS_MISSING")
    else:
        review = json.loads(summary_path.read_text())
        if review.get("both_reviewed", 0) < 600:
            problems.append("DOUBLE_REVIEW_BELOW_600")
        if review.get("intent_alpha_nominal") is None or review["intent_alpha_nominal"] < 0.80:
            problems.append("INTENT_ALPHA_BELOW_0_80")
        if review.get("unresolved_ids"):
            problems.append("UNRESOLVED_REVIEWS")
    contamination = json.loads((ROOT / "benchmarks/v5/contamination.json").read_text())
    if not contamination.get("overlap"):
        problems.append("CONTAMINATION_AUDIT_MISSING")
    return {"ready": not problems, "problems": problems, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    reviewed_path = ROOT / "benchmarks/v5/private/reviewed_manifest.jsonl"
    rows = read_jsonl(reviewed_path if reviewed_path.exists() else ROOT / "benchmarks/v5/dataset_manifest.jsonl")
    result = readiness(rows)
    print(json.dumps(result, indent=2))
    if args.require_ready and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
