"""Pin the exact private 600-row human packet before annotation begins."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
MANIFEST = ROOT / "benchmarks/v5/review_packet_manifest.json"
NAMES = ("review_packet.jsonl", "annotation_a_template.csv", "annotation_b_template.csv",
         "packet_summary.json", "candidate_disagreement.json", "review_id_map.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {name: PRIVATE / name for name in NAMES}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(name)
    count = sum(1 for line in paths["review_packet.jsonl"].open() if line.strip())
    if count != 600:
        raise ValueError(f"expected 600 review candidates, got {count}")
    summary = json.loads(paths["packet_summary.json"].read_text())
    if summary["packet_n"] != count or not summary["model_disagreement_stratified"]:
        raise ValueError("packet summary mismatch")
    state = {"version": "v5-human-packet-2", "status": "UNREVIEWED_PRIVATE_CANDIDATES",
             "n": count, "selection_seed": summary["seed"],
             "source_length_group_counts": summary["source_length_group_counts"],
             "private_file_sha256": {name: sha(path) for name, path in paths.items()},
             "candidate_manifest_sha256": sha(ROOT / "benchmarks/v5/dataset_manifest.jsonl"),
             "public_metadata_only": True}
    encoded = json.dumps(state, indent=2, sort_keys=True) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("frozen review packet changed; create a new version before annotation")
    MANIFEST.write_text(encoded)
    print(f"Verified private V5 review packet: {count} candidates")


if __name__ == "__main__":
    main()
