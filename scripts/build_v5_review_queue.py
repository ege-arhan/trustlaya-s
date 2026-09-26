"""Prepare private source-separated annotation candidates; no gold labels."""
from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from html.parser import HTMLParser

from transformers import AutoTokenizer

from trustlaya.utils import normalize
from trustlaya.v5_benchmark import bucket, normalized_hash, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
PUBLIC = ROOT / "benchmarks/v5"
TT_REV = "747a75e096761ebc01bd3970158827326b4add23"
OWASP_REV = "c04039adbe6f727a2198b3a3ea634fec98ac068a"
MICROSOFT_REV = "5b9b963b75fa2f2f4e34e459bc3647f7037b9faa"


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def plain_html(value: str) -> str:
    parser = _Text()
    parser.feed(value)
    return html.unescape(" ".join(parser.parts))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def document_paragraphs(text: str):
    section = "introduction"
    for index, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        line = paragraph.strip()
        if line.startswith("## "):
            section = line[3:].strip()
        if len(line) >= 100 and not line.startswith(("<!--", "| ---")):
            yield f"{section}:{index}", line


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v2")
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    rows, queue = [], []
    seen_normalized = set()
    duplicate_rows_skipped = 0

    def add(source: str, source_url: str, original_dataset: str, source_id: str,
            text: str, source_label: str, provenance: str, language: str,
            license_id: str, author: str | None = None) -> None:
        nonlocal duplicate_rows_skipped
        content = normalize(text)
        if not content.strip():
            return
        normalized_digest = normalized_hash(content)
        if normalized_digest in seen_normalized:
            duplicate_rows_skipped += 1
            return
        seen_normalized.add(normalized_digest)
        sample_id = f"{source}:{source_id}"
        length = len(tokenizer.backend_tokenizer.encode(content, add_special_tokens=False).ids)
        digest = hashlib.sha256(content.encode()).hexdigest()
        row = {
            "sample_id": sample_id, "source": source, "source_url": source_url,
            "original_dataset": original_dataset, "language": language,
            "provenance_type": provenance, "synthetic": False,
            "human_generated": True, "augmented": False,
            "license": license_id,
            "source_label": source_label, "review_status": "UNREVIEWED",
            "split": "CANDIDATE", "text_sha256": digest,
            "normalized_sha256": normalized_digest,
            "original_token_length": length,
            "model_visible_token_length": min(length, 94),
            "truncated": length > 94, "length_bucket": bucket(length),
            "gold_label": None, "attack_location": None,
        }
        rows.append(row)
        queue.append({"sample_id": sample_id, "text": text, "source_label": source_label,
                      "attribution_author": author,
                      "annotator_a_label": None, "annotator_b_label": None,
                      "annotator_a_location": None, "annotator_b_location": None,
                      "adjudicated_label": None, "adjudicated_location": None})

    for name, label in (("hijacking", "game_hijacking_attack_candidate"),
                        ("extraction", "game_extraction_attack_candidate")):
        filename = f"tensor_trust_{name}_v1.jsonl"
        data = PRIVATE / "source" / filename
        relative = f"benchmarks/{name}-robustness/v1/{name}_robustness_dataset.jsonl"
        for line in data.read_text().splitlines():
            item = json.loads(line)
            add("tensor_trust_game", f"https://github.com/HumanCompatibleAI/tensor-trust-data/blob/{TT_REV}/{relative}#sample-{item['sample_id']}",
                "Tensor Trust robustness v1", f"{name}:{item['sample_id']}",
                item["attack"], label, "human_game_submission_reported; per-row authorship unverified", "undetermined", "unspecified")

    doc = (PRIVATE / "source/owasp_prompt_injection_cheat_sheet.md").read_text()
    for index, (heading, text) in enumerate(document_paragraphs(doc)):
        if len(text.strip()) < 80:
            continue
        add("owasp_cheat_sheet", f"https://github.com/OWASP/CheatSheetSeries/blob/{OWASP_REV}/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.md",
            "OWASP LLM Prompt Injection Prevention Cheat Sheet", f"section:{index}:{re.sub('[^a-z0-9]+', '-', heading.lower())[:32]}",
            text, "defensive_documentation_candidate", "human_authored_document; paragraph includes code/examples", "en", "CC-BY-SA-4.0")

    doc = (PRIVATE / "source/microsoft_mcp_security.md").read_text()
    for index, (heading, text) in enumerate(document_paragraphs(doc)):
        add("microsoft_mcp_security", f"https://github.com/microsoft/mcp-for-beginners/blob/{MICROSOFT_REV}/02-Security/README.md",
            "Microsoft MCP for Beginners security lesson", f"paragraph:{index}:{heading[:24]}",
            text, "security_education_candidate", "human_authored_document_reported; per-paragraph authorship unverified", "en", "MIT")

    forum_seen = set()
    for filename in ("stackexchange_ai_prompt_injection.json", "stackexchange_so_prompt_injection.json",
                     "stackexchange_so_system_prompt.json"):
        forum = json.loads((PRIVATE / "source" / filename).read_text())
        for item in forum["items"]:
            post_id = item["question_id"]
            if post_id in forum_seen or item["creation_date"] < 1525219200:
                continue
            forum_seen.add(post_id)
            text = plain_html(item["title"] + "\n" + item.get("body", ""))
            add("stackexchange_questions", item["link"], "Stack Exchange search/advanced API 2.3",
                f"{item.get('site', filename)}:{post_id}", text,
                "human_question_candidate_unreviewed", "public_human_forum_post_reported; per-row authorship unverified",
                "undetermined", "CC-BY-SA-4.0", item.get("owner", {}).get("display_name"))

    summary = validate_manifest(rows)
    write_jsonl(PUBLIC / "dataset_manifest.jsonl", rows)
    write_jsonl(PRIVATE / "annotation_queue.jsonl", queue)
    (PUBLIC / "dataset_manifest.json").write_text(json.dumps({
        "schema_version": "v5-candidate-1", "status": "UNREVIEWED_CANDIDATES_ONLY",
        "normalized_duplicate_rows_skipped": duplicate_rows_skipped,
        "tokenizer_sha256": hashlib.sha256((ROOT / "models/trustlaya-s-v2/tokenizer.json").read_bytes()).hexdigest(),
        "effective_content_tokens": 94, "summary": summary,
        "source_count": {source: sum(r["source"] == source for r in rows) for source in sorted({r["source"] for r in rows})},
        "license_note": "Tensor Trust data repository has no explicit data license; local research candidates only; do not redistribute raw text.",
        "source_revisions": {"tensor_trust_data": TT_REV, "owasp_cheat_sheet": OWASP_REV,
                             "microsoft_mcp_security": MICROSOFT_REV,
                             "stackexchange": "API 2.3 saved response SHA-256"},
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((PRIVATE / "source").glob("*")) if path.is_file()
        },
    }, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
