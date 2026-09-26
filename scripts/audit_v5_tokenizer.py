"""Compare token lengths only; swapping tokenizer cannot evaluate model quality."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from trustlaya.utils import normalize
from trustlaya.v5_benchmark import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
REFERENCE = "google-bert/bert-base-multilingual-cased"
REFERENCE_REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    current = np.array([r["current_tokens"] for r in rows])
    reference = np.array([r["reference_tokens"] for r in rows])
    words = np.array([r["word_count"] for r in rows])
    return {"n": len(rows), "current_median_tokens": float(np.median(current)),
            "reference_median_tokens": float(np.median(reference)),
            "current_tokens_per_word": float(current.sum() / words.sum()),
            "reference_tokens_per_word": float(reference.sum() / words.sum()),
            "current_over_94": int((current > 94).sum()),
            "reference_over_94": int((reference > 94).sum()),
            "current_over_510": int((current > 510).sum()),
            "reference_over_510": int((reference > 510).sum())}


def main() -> None:
    metadata = read_jsonl(ROOT / "benchmarks/v5/dataset_manifest.jsonl")
    queue = read_jsonl(PRIVATE / "annotation_queue.jsonl")
    if [r["sample_id"] for r in metadata] != [r["sample_id"] for r in queue]:
        raise ValueError("metadata/text queue mismatch")
    current = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v2")
    current.backend_tokenizer.no_truncation()
    current.backend_tokenizer.no_padding()
    reference = AutoTokenizer.from_pretrained(REFERENCE, revision=REFERENCE_REVISION)
    reference.backend_tokenizer.no_truncation()
    reference.backend_tokenizer.no_padding()
    records = []
    for item, raw in zip(metadata, queue, strict=True):
        text = normalize(raw["text"])
        words = max(1, len(re.findall(r"\b\w+\b", text, flags=re.UNICODE)))
        records.append({"source": item["source"], "word_count": words,
                        "current_tokens": len(current.backend_tokenizer.encode(text, add_special_tokens=False).ids),
                        "reference_tokens": len(reference.backend_tokenizer.encode(text, add_special_tokens=False).ids)})
    output = {"status": "TOKEN_LENGTH_AUDIT_ONLY", "candidate_n": len(records),
              "current_model": "ytu-ce-cosmos/turkish-medium-bert-uncased",
              "current_tokenizer_sha256": hashlib.sha256((ROOT / "models/trustlaya-s-v2/tokenizer.json").read_bytes()).hexdigest(),
              "reference_tokenizer": REFERENCE, "reference_revision": REFERENCE_REVISION,
              "reference_license": "Apache-2.0", "by_source": {
                  source: summarize([r for r in records if r["source"] == source])
                  for source in sorted({r["source"] for r in records})},
              "warning": "Token counts do not prove accuracy or permit replacement of the trained tokenizer."}
    (ROOT / "benchmarks/v5/tokenizer_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
