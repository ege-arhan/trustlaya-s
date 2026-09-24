"""Audit the published synthetic splits without changing them."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from trustlaya.dataset import read_rows
from trustlaya.labels import TASKS

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def normalized(text):
    text = text.casefold()
    text = re.sub(r"[\w.+-]+@[\w.-]+", " EMAIL ", text)
    text = re.sub(r"\b(?:https?://)?[\w.-]+://\S+", " CREDENTIAL ", text)
    text = re.sub(r"\d+", "0", text)
    return " ".join(text.split())


def main():
    split_rows = {name: read_rows(ROOT / f"data/splits/{name}.jsonl") for name in SPLITS}
    all_rows = [(name, row) for name, rows in split_rows.items() for row in rows]
    texts = [row["text"] for _, row in all_rows]
    norms = [normalized(text) for text in texts]
    split_sets = {
        name: {
            "exact": {row["text"] for row in rows},
            "normalized": {normalized(row["text"]) for row in rows},
            "family": {row["family"] for row in rows},
        }
        for name, rows in split_rows.items()
    }
    pairwise = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            pairwise[f"{a}-{b}"] = {
                key + "_overlap": len(split_sets[a][key] & split_sets[b][key])
                for key in ("exact", "normalized", "family")
            }

    # Only 143 normalized text prototypes exist in this corpus. Full cross-split
    # prototype similarity is cheap and every original row maps to a prototype.
    prototypes = sorted(set(norms))
    lookup = {text: i for i, text in enumerate(prototypes)}
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    vectors = vectorizer.fit_transform(prototypes)
    proto_splits = defaultdict(set)
    for name, row in all_rows:
        proto_splits[normalized(row["text"])].add(name)
    comparison = {}
    for name in SPLITS:
        own = [p for p in prototypes if name in proto_splits[p]]
        other = [p for p in prototypes if any(s != name for s in proto_splits[p])]
        cosine = (vectors[[lookup[x] for x in own]] @ vectors[[lookup[x] for x in other]].T).toarray()
        token_sets = [set(re.findall(r"\w+", p)) for p in prototypes]
        max_by_proto = {}
        for i, p in enumerate(own):
            j = int(cosine[i].argmax())
            a, b = token_sets[lookup[p]], token_sets[lookup[other[j]]]
            max_by_proto[p] = {
                "cosine": float(cosine[i, j]),
                "token_jaccard": len(a & b) / max(1, len(a | b)),
                "nearest_other_prototype": other[j],
            }
        row_scores = [max_by_proto[normalized(r["text"])] for r in split_rows[name]]
        comparison[name] = {
            "rows_cosine_ge_0_80_to_other_split": sum(x["cosine"] >= 0.80 for x in row_scores),
            "rows_cosine_ge_0_90_to_other_split": sum(x["cosine"] >= 0.90 for x in row_scores),
            "rows_token_jaccard_ge_0_80_to_nearest_cosine_match": sum(x["token_jaccard"] >= 0.80 for x in row_scores),
            "max_cross_split_cosine": max(x["cosine"] for x in row_scores),
        }

    by_split = {}
    for name, rows in split_rows.items():
        counts = Counter(row["category"] for row in rows)
        by_split[name] = {
            "rows": len(rows),
            "exact_duplicates_within_split": len(rows) - len({r["text"] for r in rows}),
            "normalized_duplicates_within_split": len(rows) - len({normalized(r["text"]) for r in rows}),
            "languages": dict(sorted(Counter(r["language"] for r in rows).items())),
            "categories": dict(sorted(counts.items())),
            "labels_positive": {task: sum(r["labels"][task] for r in rows) for task in TASKS},
            "families": len({r["family"] for r in rows}),
            "template_families": dict(sorted(Counter(r["family"] for r in rows).items())),
        }
    report = {
        "total_rows": len(all_rows), "unique_exact_texts": len(set(texts)),
        "unique_normalized_texts": len(set(norms)),
        "exact_duplicates_all_rows": len(texts) - len(set(texts)),
        "normalized_duplicates_all_rows": len(norms) - len(set(norms)),
        "splits": by_split, "cross_split": pairwise,
        "near_duplicate_proxy": comparison,
        "method": "Lowercase, replace emails and digit runs, collapse whitespace; character TF-IDF cosine (3-5 grams) and token Jaccard on normalized prototypes. Threshold counts are row counts, not independent examples. This does not prove or disprove semantic paraphrase leakage.",
    }
    (ROOT / "reports/dataset_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lines = ["# Synthetic dataset audit", "", "Audited the unchanged 10,000-row local splits. Source: controlled templates only; no independently adjudicated labels.", "", f"Exact unique texts: {report['unique_exact_texts']:,}; normalized unique prototypes: {report['unique_normalized_texts']:,}. Exact repeated rows: {report['exact_duplicates_all_rows']:,}; normalized repeated rows: {report['normalized_duplicates_all_rows']:,}.", "", "| Split | Rows | Families | Turkish | English | Mixed | Exact repeated | Normalized repeated |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in by_split.items():
        languages = row["languages"]
        lines.append(f"| {name} | {row['rows']} | {row['families']} | {languages.get('tr', 0)} | {languages.get('en', 0)} | {languages.get('mixed', 0)} | {row['exact_duplicates_within_split']} | {row['normalized_duplicates_within_split']} |")
    lines += ["", "**Critical split confound:** all original test rows are labeled mixed-language; validation is English-only; training contains Turkish and English but no mixed rows. Thus original held-out scores cannot be reported as separate Turkish or English test performance.", "", "| Split pair | Exact text overlap | Normalized overlap | Family overlap |", "|---|---:|---:|---:|"]
    for pair, row in pairwise.items():
        lines.append(f"| {pair} | {row['exact_overlap']} | {row['normalized_overlap']} | {row['family_overlap']} |")
    lines += ["", "Near-duplicate proxy using character TF-IDF cosine on normalized templates:", "", "| Split | Rows cosine ≥ 0.80 to another split | Rows cosine ≥ 0.90 | Rows token Jaccard ≥ 0.80 |", "|---|---:|---:|---:|"]
    for name, row in comparison.items():
        lines.append(f"| {name} | {row['rows_cosine_ge_0_80_to_other_split']} | {row['rows_cosine_ge_0_90_to_other_split']} | {row['rows_token_jaccard_ge_0_80_to_nearest_cosine_match']} |")
    lines += ["", "These similarity counts flag possible shared phrasing; they are not manual paraphrase labels. Family-disjoint splitting removes identical template families but cannot remove semantic overlap among different templates. Repeated literals and small template pool create strong memorization risk.", "", "## Label distribution", "", "| Task | Train positive | Validation positive | Test positive | Total positive |", "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        vals = [by_split[s]["labels_positive"][task] for s in SPLITS]
        lines.append(f"| {task} | {vals[0]} | {vals[1]} | {vals[2]} | {sum(vals)} |")
    lines += ["", "Category, language and all template-family counts are in `reports/dataset_audit.json`. The new Generalization and TR benchmarks should allocate held-out families separately within each language and independently label external cases."]
    (ROOT / "reports/dataset_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"rows": report["total_rows"], "unique_exact": report["unique_exact_texts"], "unique_normalized": report["unique_normalized_texts"], "cross_split": pairwise, "near_duplicate_proxy": comparison}, indent=2))


if __name__ == "__main__":
    main()
