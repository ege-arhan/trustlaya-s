"""Create a source-separated, local-only diagnostic DEV cohort; never read final test."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "benchmarks/external/predictions"
OWASP = ROOT / "benchmarks/external/raw/v4/owasp"
BORD_REV = "398e3f875ffd32ec2e6817a95feebfcbae41643a"
OWASP_REV = "9253e38ade58e959b531c0c5c9a4842272c9cd0e"
# Char 4-5-gram cosine 0.959 match to frozen JLL test; excluded before
# final threshold/strategy reselection. See reports/v4_data_leakage.json.
EXCLUDE_IDS = {"bordair:13198"}


def main():
    import subprocess
    assert subprocess.check_output(["git", "-C", str(OWASP), "rev-parse", "HEAD"], text=True).strip() == OWASP_REV
    path = Path(hf_hub_download("Bordair/bordair-multimodal", "payloads_live/attacks.jsonl",
                                repo_type="dataset", revision=BORD_REV))
    rows = []
    seen = set()
    for i, line in enumerate(path.open()):
        item = json.loads(line)
        text = item["text"].strip()
        # Live submissions can be uninformative in isolation (e.g. "hello").
        # This length filter is fixed without looking at model predictions.
        if len(text) < 500: continue
        digest = hashlib.sha256(re.sub(r"\s+", " ", text.casefold()).encode()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        if f"bordair:{i}" in EXCLUDE_IDS: continue
        rows.append({"id": f"bordair:{i}", "text": text, "gold": 1,
                     "label": "GAME_INJECTION_ATTEMPT", "source": "Bordair live game",
                     "sha256": digest, "license": "MIT", "language": "unannotated"})
    attacks = len(rows)
    for file in sorted((OWASP / "2026/final").glob("LLM*.md")):
        for i, paragraph in enumerate(re.split(r"\n\s*\n", file.read_text())):
            text = re.sub(r"\s+", " ", paragraph).strip()
            if len(text.split()) < 40 or not re.search(r"prompt inject|jailbreak|security|attack|system prompt", text, re.I):
                continue
            digest = hashlib.sha256(text.casefold().encode()).hexdigest()
            if digest in seen: continue
            seen.add(digest)
            rows.append({"id": f"owasp:{file.stem}:{i}", "text": text, "gold": 0,
                         "label": "SECURITY_DOCUMENTATION", "source": "OWASP LLM Top 10 2026",
                         "sha256": digest, "license": "CC BY-SA 4.0", "language": "English"})
    LOCAL.mkdir(parents=True, exist_ok=True)
    output = LOCAL / "v4_dev_text_private.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "attack": attacks, "benign_docs": len(rows)-attacks,
                      "Bordair_revision": BORD_REV, "OWASP_revision": OWASP_REV,
                      "near_test_excluded_ids": sorted(EXCLUDE_IDS)}, indent=2))


if __name__ == "__main__": main()
