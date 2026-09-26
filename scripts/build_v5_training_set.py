"""Build the V5 multi-source attack/benign dataset. No training happens here.

Sources are fetched at pinned revisions with SHA-256 checks (data/v5_source_pins.json)
into git-ignored data/external/. Text splits go to git-ignored data/v5/. Only
hashes, labels, splits and statistics are written to tracked files:

  data/v5_manifest.json, data/v5_manifest.sha256, data/label_mapping.json,
  data/dataset_lineage.json, reports/dataset_dedup_report.json,
  reports/tensor_trust_training_stats.json

Labels are source-native: nobody annotates rows and no LLM labels are used.
Near-duplicates (char 5-gram MinHash/LSH, verified Jaccard > 0.70) are merged
into clusters and every cluster goes to exactly one split, across sources.

Run: .venv/bin/python scripts/build_v5_training_set.py [--record]
"""

import argparse
import bz2
import hashlib
import html
import json
import random
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_tensor_trust import CACHE as TT_CACHE, ROOT, fetch_pinned, main as fetch_tt  # noqa: E402
from trustlaya.tensor_trust import (LABELS, derive, native_category, parse_attack,  # noqa: E402
                                    parse_benchmark_row)

OUT = ROOT / "data/v5"
EXTERNAL = ROOT / "data/external"
SEED = 42
JACCARD = 0.70
SHINGLE = 5
PERMUTATIONS = 128
SPLITS = (("TRAIN", 70), ("DEV", 15), ("TEST", 15))
HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
GITHUB = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"
ARXIV_QUERIES = {
    "prompt_injection": 'abs:"prompt injection"',
    "jailbreak": 'abs:jailbreak AND cat:cs.CL',
}
# Pool sources are split into TRAIN/DEV/TEST; OOD sources are test-only.
POOL = ("tensor_trust", "jailbreakllms", "security_docs", "arxiv_abstracts")
OOD = ("deepset", "gandalf", "jailbreakbench")
CUE = re.compile(r"(?i)\b(?:ignore|disregard|forget|override|bypass|reveal|pretend|jailbreak|"
                 r"system prompt|previous instructions|you are now|DAN)\b")


def normalize(text):
    return re.sub(r"\s+", " ", text.casefold()).strip()


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hf(source, repo, rev, paths, record):
    return fetch_pinned(source, f"https://huggingface.co/datasets/{repo}", rev,
                        {p: HF.format(repo=repo, rev=rev, path=p) for p in paths},
                        EXTERNAL / source / rev, record=record)


class _Text(HTMLParser):
    """HTML to text, dropping <pre>, <code> and <blockquote> (quoted payloads)."""

    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("pre", "code", "blockquote"):
            self.skip += 1
        elif tag in ("p", "br", "li", "h1", "h2", "h3"):
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("pre", "code", "blockquote") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_text(value):
    parser = _Text()
    parser.feed(html.unescape(value))
    return "".join(parser.parts)


def prose_paragraphs(text):
    """Documentation/forum prose only: no fenced code, quotes, tables or lone payload lines."""
    text = re.sub(r"```.*?```", "\n\n", text, flags=re.S)
    kept, dropped = [], Counter()
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if len(block) < 80:
            dropped["short"] += 1
        elif block.startswith((">", "|", "    ", "#")):
            dropped["quote_table_or_heading"] += 1
        elif re.search(r"[\"“'`][^\"”'`]{0,200}\b(?:ignore|disregard|forget|you are now)\b", block, re.I):
            dropped["quoted_attack_string"] += 1
        elif re.match(r"(?i)^(?:ignore|disregard|forget|you are now|pretend|act as)\b", block):
            dropped["imperative_payload_like"] += 1
        else:
            kept.append(re.sub(r"\s+", " ", block))
    return kept, dropped


def row(source, native_id, text, native, label, *, license, provenance, language="undetermined",
        extra=None):
    return {"sample_id": f"{source}:{native_id}", "source": source, "native_category": native,
            "label": label, "text": text, "license": license, "provenance": provenance,
            "language": language, **(extra or {})}


def load_tensor_trust(stats):
    """Text-level categories from the v2 raw dump; v1 used to validate the parser."""
    bench = {}
    for kind in ("hijacking", "extraction"):
        path = TT_CACHE / f"benchmarks/{kind}-robustness/v1/{kind}_robustness_dataset.jsonl"
        bench[kind] = {parse_benchmark_row(json.loads(line)) for line in path.open()}
    validation = {}
    for version in ("v1", "v2"):
        path = TT_CACHE / f"raw-data/{version}/raw_dump_attacks.jsonl.bz2"
        attacks = [parse_attack(json.loads(line)) for line in bz2.open(path, "rt")]
        flags = derive(attacks)
        hijack = {a["input"] for a in attacks if flags[a["id"]] and flags[a["id"]]["hijacking_candidate"]}
        extract = {a["input"] for a in attacks if flags[a["id"]] and flags[a["id"]]["extraction_candidate"]}
        validation[version] = {
            "raw_attacks": len(attacks),
            "hijacking_benchmark_unique": len(bench["hijacking"]),
            "hijacking_benchmark_recovered": len(bench["hijacking"] & hijack),
            "extraction_benchmark_unique": len(bench["extraction"]),
            "extraction_benchmark_recovered": len(bench["extraction"] & extract),
            "hijacking_candidates_unique": len(hijack),
            "extraction_candidates_unique": len(extract),
        }
    priority = list(LABELS)  # benchmark > candidates > excluded categories
    best, first_id, occurrences, per_row = {}, {}, Counter(), Counter()
    for attack in attacks:  # v2, the superset dump
        category = native_category(flags[attack["id"]], attack["input"],
                                   bench["hijacking"], bench["extraction"])
        per_row[category] += 1
        text = attack["input"]
        if text is None or not text.strip():
            continue
        occurrences[text] += 1
        if text not in best or priority.index(category) < priority.index(best[text]):
            best[text] = category
            first_id[text] = attack["id"]
    stats.update(parser_validation=validation, raw_rows_by_category=dict(per_row),
                 unique_texts_by_category=dict(Counter(best.values())))
    return [row("tensor_trust", first_id[t], t, c, LABELS[c],
                license="unspecified (data repo has no LICENSE file; game code BSD-2-Clause)",
                provenance="human game submissions (Tensor Trust raw dump v2, 2024-02-16)",
                extra={"raw_occurrences": occurrences[t],
                       "upstream_benchmark": c.endswith("benchmark_v1")})
            for t, c in best.items()]


def load_jailbreakllms(record):
    rev = "a10aab8eff1c73165a442d4464dce192bd28b9c5"
    paths = hf("jailbreakllms", "TrustAIRLab/in-the-wild-jailbreak-prompts", rev,
               ["jailbreak_2023_12_25/train-00000-of-00001.parquet",
                "regular_2023_12_25/train-00000-of-00001.parquet", "README.md"], record)
    rows = []
    for key, native in (("jailbreak_2023_12_25/train-00000-of-00001.parquet", "jailbreak_prompt"),
                        ("regular_2023_12_25/train-00000-of-00001.parquet", "regular_prompt")):
        frame = pd.read_parquet(paths[key])
        expected = {"jailbreak_prompt": True, "regular_prompt": False}[native]
        if set(frame["jailbreak"].unique()) != {expected}:
            raise ValueError(f"JailbreakLLMs {native} has unexpected jailbreak flags")
        for index, item in frame.iterrows():
            rows.append(row("jailbreakllms", f"{native}:{index}", item["prompt"], native,
                            "ATTACK" if expected else "BENIGN", license="MIT",
                            provenance=f"in-the-wild prompts ({item['platform']}), author-labeled",
                            extra={"platform": item["platform"]}))
    return rows


def load_ood(record):
    rows = []
    rev = "4f61ecb038e9c3fb77e21034b22511b523772cdd"
    files = ["data/train-00000-of-00001-9564e8b05b4757ab.parquet",
             "data/test-00000-of-00001-701d16158af87368.parquet"]
    paths = hf("deepset", "deepset/prompt-injections", rev, files + ["README.md"], record)
    for split, key in zip(("train", "test"), files):
        for index, item in pd.read_parquet(paths[key]).iterrows():
            if item["label"] not in (0, 1):
                raise ValueError("deepset label outside {0,1}")
            rows.append(row("deepset", f"{split}:{index}", item["text"],
                            "injection" if item["label"] else "benign",
                            "ATTACK" if item["label"] else "BENIGN", license="Apache-2.0",
                            provenance="deepset curated set; generation method not documented per row",
                            language="mixed (en/de)"))
    rev = "04737b65e90a6794ec227012e4a255a7def6344b"
    files = ["data/train-00000-of-00001-ded53be747ff55cd.parquet",
             "data/validation-00000-of-00001-94481a2a09ff2fff.parquet",
             "data/test-00000-of-00001-bc92128b9288a6d1.parquet"]
    paths = hf("gandalf", "Lakera/gandalf_ignore_instructions", rev, files + ["README.md"], record)
    for key in files:
        for index, item in pd.read_parquet(paths[key]).iterrows():
            rows.append(row("gandalf", f"{key.split('/')[1].split('-')[0]}:{index}", item["text"],
                            "gandalf_ignore_instructions_attack", "ATTACK", license="MIT",
                            provenance="Gandalf game submissions filtered by Lakera"))
    rev = "886acc352a31533ffbcf4ef22c744658688086fc"
    paths = hf("jailbreakbench", "JailbreakBench/JBB-Behaviors", rev,
               ["data/harmful-behaviors.csv", "data/benign-behaviors.csv", "README.md", "LICENSE"], record)
    for key, native, label in (("data/harmful-behaviors.csv", "harmful_behavior", "HARMFUL_REQUEST"),
                               ("data/benign-behaviors.csv", "benign_behavior", "BENIGN_REQUEST")):
        for _, item in pd.read_csv(paths[key]).iterrows():
            rows.append(row("jailbreakbench", f"{native}:{item['Index']}", item["Goal"], native, label,
                            license="MIT", language="en",
                            provenance=f"JBB-Behaviors goal ({item['Source']}); a request, not a jailbreak prompt"))
    return rows


def load_security_docs(record, stats):
    rows, dropped = [], Counter()
    docs = {
        "owasp_cheat_sheet": ("OWASP/CheatSheetSeries", "c04039adbe6f727a2198b3a3ea634fec98ac068a",
                              "cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.md", "CC-BY-SA-4.0"),
        "microsoft_mcp_security": ("microsoft/mcp-for-beginners", "5b9b963b75fa2f2f4e34e459bc3647f7037b9faa",
                                   "02-Security/README.md", "MIT"),
    }
    for name, (repo, rev, path, license) in docs.items():
        paths = fetch_pinned(f"security_docs:{name}", f"https://github.com/{repo}", rev,
                             {path: GITHUB.format(repo=repo, rev=rev, path=path)},
                             EXTERNAL / "security_docs" / name / rev, record=record)
        kept, lost = prose_paragraphs(paths[path].read_text())
        dropped.update(lost)
        rows += [row("security_docs", f"{name}:{i}", text, "defensive_documentation", "BENIGN",
                     license=license, language="en", provenance=f"{repo}@{rev[:7]} prose paragraph")
                 for i, text in enumerate(kept)]
    # Stack Exchange API responses saved on 2026-09-25; not re-downloadable byte-for-byte.
    saved = ROOT / "benchmarks/v5/private/source"
    names = ("stackexchange_ai_prompt_injection.json", "stackexchange_so_prompt_injection.json",
             "stackexchange_so_system_prompt.json")
    paths = fetch_pinned("security_docs:stackexchange", "https://api.stackexchange.com/2.3 (saved)",
                         "saved-2026-09-25", {n: (saved / n).as_uri() for n in names},
                         EXTERNAL / "security_docs/stackexchange", record=record)
    seen = set()
    for name in names:
        for item in json.loads(paths[name].read_text())["items"]:
            if item["question_id"] in seen:
                continue
            seen.add(item["question_id"])
            kept, lost = prose_paragraphs(html_text(item.get("body", "")))
            dropped.update(lost)
            rows += [row("security_docs", f"stackexchange:{item['question_id']}:{i}", text,
                         "forum_question_prose", "BENIGN", license=item.get("content_license", "CC BY-SA"),
                         language="en", provenance="Stack Exchange question body, code/quotes removed")
                     for i, text in enumerate(kept)]
    stats["security_docs_dropped_blocks"] = dict(dropped)
    return rows


def load_arxiv(record):
    rows = []
    for name, query in ARXIV_QUERIES.items():
        url = ("https://export.arxiv.org/api/query?" +
               urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": 300,
                                       "sortBy": "submittedDate", "sortOrder": "ascending"}, safe=":"))
        path = fetch_pinned(f"arxiv:{name}", "https://export.arxiv.org/api", "query-2026-09-26",
                            {f"{name}.atom": url}, EXTERNAL / "arxiv", record=record)[f"{name}.atom"]
        feed = path.read_text()
        for entry in re.findall(r"<entry>(.*?)</entry>", feed, flags=re.S):
            arxiv_id = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", entry).group(1)
            summary = html.unescape(re.search(r"<summary>(.*?)</summary>", entry, re.S).group(1))
            rows.append(row("arxiv_abstracts", arxiv_id, re.sub(r"\s+", " ", summary).strip(),
                            "paper_abstract", "BENIGN", license="arXiv metadata (CC0 1.0)",
                            language="en", provenance="author-written abstract via arXiv API"))
    return rows


def cluster(rows):
    """Union-find over exact-normalized and MinHash near-duplicates (verified Jaccard)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    by_norm = {}
    for i, item in enumerate(rows):
        key = item["normalized_sha256"]
        if key in by_norm:
            union(i, by_norm[key])
        else:
            by_norm[key] = i
    shingles = []
    lsh = MinHashLSH(threshold=JACCARD, num_perm=PERMUTATIONS)
    for i, item in enumerate(rows):
        doc = item["text"].lower().strip().replace("\r", "").replace("\n", "")
        grams = {doc[k:k + SHINGLE].encode() for k in range(max(1, len(doc) - SHINGLE + 1))}
        shingles.append(grams)
        signature = MinHash(num_perm=PERMUTATIONS, seed=SEED)
        signature.update_batch(list(grams))
        for j in lsh.query(signature):
            if len(grams & shingles[j]) / len(grams | shingles[j]) > JACCARD:
                union(i, j)
        lsh.insert(i, signature)
    return [find(i) for i in range(len(rows))]


def assign(rows, roots):
    members = defaultdict(list)
    for i, root in enumerate(roots):
        members[root].append(i)
    report = Counter()
    for root, idx in members.items():
        group = [rows[i] for i in idx]
        sources = {r["source"] for r in group}
        key = min(r["normalized_sha256"] for r in group)
        pool = [r for r in group if r["source"] in POOL]
        if any(r.get("upstream_benchmark") for r in pool):
            split = "TEST"  # upstream Tensor Trust benchmark rows stay out of TRAIN/DEV
        else:
            bucket = int(key[:8], 16) % 100
            split = next(name for name, share in _cumulative() if bucket < share)
        labels = {r["label"] for r in group if r["label"] in ("ATTACK", "BENIGN")}
        for r in group:
            r["cluster"] = key[:16]
            r["cluster_size"] = len(group)
            r["cluster_sources"] = sorted(sources)
            r["split"] = split if r["source"] in POOL else "OOD_TEST"
            r["status"] = "included"
            if r["label"] == "EXCLUDED":
                r["status"] = "excluded_source_semantics"
            elif labels == {"ATTACK", "BENIGN"} and r["label"] == "BENIGN":
                r["status"] = "excluded_near_duplicate_of_attack"
            elif r["source"] in OOD and pool:
                r["status"] = "excluded_ood_overlaps_pool"
            report[(r["source"], r["status"])] += 1
        if len(sources) > 1:
            report[("cross_source_cluster", "+".join(sorted(sources)))] += 1
    return report


def _cumulative():
    total = 0
    for name, share in SPLITS:
        total += share
        yield name, total


def training_variants(rows):
    train = [r for r in rows if r["split"] == "TRAIN" and r["status"] == "included"]
    attacks = defaultdict(list)
    for r in train:
        if r["label"] == "ATTACK":
            attacks[r["source"]].append(r)
    benign = [r for r in train if r["label"] == "BENIGN"]
    rng = random.Random(SEED)
    # B caps Tensor Trust at twice all other attack sources together (declared rule).
    others = sum(len(v) for k, v in attacks.items() if k != "tensor_trust")
    capped = rng.sample(attacks["tensor_trust"], min(len(attacks["tensor_trust"]), 2 * others))
    floor = min(len(v) for v in attacks.values())
    balanced = [x for v in attacks.values() for x in rng.sample(v, floor)]
    variants = {
        "A_natural": [x for v in attacks.values() for x in v] + benign,
        "B_capped_tensor_trust": capped + [x for k, v in attacks.items() if k != "tensor_trust" for x in v] + benign,
        "C_balanced_source": balanced + benign,
    }
    summary = {}
    for name, items in variants.items():
        count = Counter((r["source"], r["label"]) for r in items)
        summary[name] = {"n": len(items), "by_source_label": {f"{s}/{l}": c for (s, l), c in sorted(count.items())},
                         "attack_share_tensor_trust": round(
                             count[("tensor_trust", "ATTACK")] / max(1, sum(c for (s, l), c in count.items() if l == "ATTACK")), 4)}
    return variants, summary


EXPERIMENTS = {
    "A": {"train": {"tensor_trust", "jailbreakllms"}, "dev": {"tensor_trust", "jailbreakllms"},
          "test": {"deepset", "gandalf"}, "note": "Test on independent sources never used for training."},
    "B": {"train": {"jailbreakllms"}, "dev": {"jailbreakllms"}, "test": {"tensor_trust"},
          "note": "Tensor Trust TEST split is attack-only: recall is defined, FPR is not."},
    "C": {"train": {"tensor_trust", "security_docs", "arxiv_abstracts"},
          "dev": {"tensor_trust", "security_docs", "arxiv_abstracts"}, "test": {"jailbreakllms"},
          "note": "Tensor Trust has no benign rows; negatives come from public security prose."},
    "D": {"train": set(POOL), "dev": set(POOL), "test": {"deepset", "gandalf", "jailbreakbench"},
          "note": "JailbreakBench measures harmful requests, a different task; reported separately."},
}


def experiment_ids(rows):
    out = {}
    for name, spec in EXPERIMENTS.items():
        role = lambda r, sources, split: (r["status"] == "included" and r["source"] in sources
                                          and r["split"] == split)
        test_split = lambda r: "TEST" if r["source"] in POOL else "OOD_TEST"
        out[name] = {
            "train": sorted(r["sample_id"] for r in rows if role(r, spec["train"], "TRAIN")),
            "dev": sorted(r["sample_id"] for r in rows if role(r, spec["dev"], "DEV")),
            "test": sorted(r["sample_id"] for r in rows if role(r, spec["test"], test_split(r))),
            "sources": {k: sorted(v) for k, v in spec.items() if k != "note"}, "note": spec["note"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="pin sources that have no checksum yet")
    args = parser.parse_args()
    sys.argv = [sys.argv[0]] + (["--record"] if args.record else [])
    fetch_tt()
    tt_stats = {}
    rows = load_tensor_trust(tt_stats)
    rows += load_jailbreakllms(args.record) + load_ood(args.record)
    doc_stats = {}
    rows += load_security_docs(args.record, doc_stats) + load_arxiv(args.record)
    for item in rows:
        item["text_sha256"] = sha(item["text"])
        item["normalized_sha256"] = sha(normalize(item["text"]))
    # Exact (normalized) duplicates within a source collapse to one row. Usable
    # labels win over EXCLUDED variants; ATTACK vs BENIGN on the same text is a
    # conflict and both are dropped.
    seen, exact_dups = {}, Counter()
    rank = {"ATTACK": 0, "BENIGN": 0, "HARMFUL_REQUEST": 0, "BENIGN_REQUEST": 0, "EXCLUDED": 1}
    for item in rows:
        key = (item["source"], item["normalized_sha256"])
        kept = seen.get(key)
        if kept is None:
            seen[key] = item
            continue
        exact_dups[item["source"]] += 1
        if {kept["label"], item["label"]} == {"ATTACK", "BENIGN"}:
            kept["label_conflict_exact"] = True
        elif rank[item["label"]] < rank[kept["label"]] or (
                item["label"] == kept["label"] and item.get("upstream_benchmark") and not kept.get("upstream_benchmark")):
            item["label_conflict_exact"] = kept.get("label_conflict_exact", False)
            seen[key] = item
    rows = list(seen.values())
    for item in rows:
        if item.pop("label_conflict_exact", False):
            item["label"] = "EXCLUDED"
            item["native_category"] += "+exact_label_conflict"
    included = [r for r in rows if r["label"] != "EXCLUDED"]
    roots = cluster(included)
    status = assign(included, roots)
    for r in rows:
        if r["label"] == "EXCLUDED":
            r.update(split="NONE", status="excluded_source_semantics", cluster=None)
    variants, variant_summary = training_variants(included)
    experiments = experiment_ids(included)

    # Leakage audit: no cluster may appear in two splits of the pool.
    cluster_splits = defaultdict(set)
    for r in included:
        if r["status"] == "included":
            cluster_splits[r["cluster"]].add(r["split"])
    leaking = sorted(c for c, s in cluster_splits.items()
                     if len(s - {"OOD_TEST"}) > 1 or ("OOD_TEST" in s and len(s) > 1))
    counts = Counter((r["source"], r["split"], r["label"]) for r in included if r["status"] == "included")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in variants.items():
        with open(OUT / f"train_{name}.jsonl", "w") as out:
            for r in items:
                out.write(json.dumps({k: r[k] for k in ("sample_id", "source", "label", "text")},
                                     ensure_ascii=False) + "\n")
    with open(OUT / "all_rows.jsonl", "w") as out:
        for r in included:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    lineage_fields = ["sample_id", "source", "native_category", "label", "split", "status",
                      "cluster", "cluster_size", "text_sha256_16", "normalized_sha256", "char_length"]
    for r in rows:
        r["char_length"] = len(r["text"])
        r["text_sha256_16"] = r["text_sha256"][:16]
    lineage = {"fields": lineage_fields,
               "rows": [[r.get(f) for f in lineage_fields] for r in
                        sorted(rows, key=lambda r: r["sample_id"])
                        if r["source"] != "tensor_trust" or r["label"] != "EXCLUDED"],
               "note": "Tensor Trust EXCLUDED texts are counted in reports/tensor_trust_training_stats.json, not listed.",
               "experiments": experiments}
    (ROOT / "data/dataset_lineage.json").write_text(json.dumps(lineage, separators=(",", ":")) + "\n")

    mapping = {
        "tensor_trust": {k: {"label": v, "rule": {
            "hijacking_benchmark_v1": "attacker_input appears in upstream v1 hijacking robustness benchmark (LLM adversarially filtered)",
            "extraction_benchmark_v1": "attacker_input appears in upstream v1 extraction robustness benchmark",
            "hijacking_candidate": "upstream is_first_pass_hijacking_candidate: granted access without being the access code, used against >=2 accounts, >1 word or >30 chars",
            "extraction_candidate": "upstream: response contains access code, or last failed attack before the attacker entered the code (not self-attack)",
            "unverified_attempt": "attack on another player with no success evidence; NOT benign",
            "self_or_sandbox_attack": "attacker == defender (testing own defense / sandbox)",
            "access_code_entry": "input equals the defender access code (a password, not an attack)",
            "tos_removed_text": "input or access code removed upstream for ToS violations"}[k]}
            for k, v in LABELS.items()},
        "tensor_trust_defenses": {"label": "EXCLUDED", "rule": "defense prompts are not a normal-user distribution"},
        "jailbreakllms": {"jailbreak_prompt": "ATTACK", "regular_prompt": "BENIGN"},
        "deepset": {"injection": "ATTACK", "benign": "BENIGN"},
        "gandalf": {"gandalf_ignore_instructions_attack": "ATTACK"},
        "jailbreakbench": {"harmful_behavior": "HARMFUL_REQUEST (different task)",
                           "benign_behavior": "BENIGN_REQUEST (different task)"},
        "security_docs": {"defensive_documentation": "BENIGN", "forum_question_prose": "BENIGN",
                          "filter": "code blocks, quotes, tables, quoted or imperative attack strings removed"},
        "arxiv_abstracts": {"paper_abstract": "BENIGN"},
        "conflicts": {"exact_label_conflict": "EXCLUDED",
                      "benign near-duplicate of an attack (Jaccard > 0.70)": "excluded"},
    }
    (ROOT / "data/label_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")

    pins = json.loads((ROOT / "data/v5_source_pins.json").read_text())
    sources = {}
    source_meta = {
        "tensor_trust": ("TRAIN/DEV/TEST", "prompt hijacking + extraction attacks", "undetermined (mostly English)"),
        "jailbreakllms": ("TRAIN/DEV/TEST", "jailbreak vs regular prompts", "undetermined (mostly English)"),
        "security_docs": ("TRAIN/DEV/TEST", "benign security prose (hard negatives)", "en"),
        "arxiv_abstracts": ("TRAIN/DEV/TEST", "benign security prose (hard negatives)", "en"),
        "deepset": ("OOD_TEST", "prompt injection vs benign", "mixed en/de"),
        "gandalf": ("OOD_TEST", "prompt injection (attack only)", "undetermined (mostly English)"),
        "jailbreakbench": ("OOD_TEST", "harmful vs benign requests (task mismatch)", "en"),
    }
    for source, (roles, task, language) in source_meta.items():
        pinned = {k: v for k, v in pins.items() if k == source or k.startswith(source + ":")}
        items = [r for r in rows if r["source"] == source]
        sources[source] = {
            "pins": pinned, "roles": roles, "intended_task": task, "language": language,
            "license": sorted({r["license"] for r in items}),
            "provenance": sorted({re.sub(r"\(.*?\)", "", r["provenance"]).strip() for r in items})[:5],
            "rows_loaded": len(items),
            "label_schema": sorted({f"{r['native_category']}->{r['label']}" for r in items})[:12],
            "usable_for_training": source in POOL,
            "usable_for_test": True,
            "counts": {f"{sp}/{lb}": c for (s, sp, lb), c in sorted(counts.items()) if s == source},
        }
    sources["tab"] = {"pins": {"repo": "NorskRegnesentral/text-anonymization-benchmark",
                               "revision": "558e09e26d6b36f5f78440074e6a233946d98bd9"},
                      "roles": "separate PII task (exact-span F1)", "intended_task": "PII span detection",
                      "language": "en", "license": ["MIT"], "usable_for_training": True,
                      "usable_for_test": True, "note": "Not mixed into attack classification; see reports/external_real_world_benchmark.md."}
    sources["hackaprompt"] = {"status": "not_included",
                              "reason": "gated on Hugging Face (terms must be accepted by the account owner)"}
    manifest = {
        "schema": "v5-multisource-1", "seed": SEED,
        "dedup": {"normalization": "casefold + whitespace collapse (exact)",
                  "near_duplicate": f"MinHash LSH, char {SHINGLE}-grams, {PERMUTATIONS} permutations, verified Jaccard > {JACCARD}",
                  "split_unit": "cluster (cross-source)", "split_shares": dict(SPLITS)},
        "sources": sources,
        "training_variants": variant_summary,
        "default_training_variant": "C_balanced_source (declared before any training or test result)",
        "leaking_clusters": len(leaking),
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (ROOT / "data/v5_manifest.json").write_text(manifest_text)
    lineage_bytes = (ROOT / "data/dataset_lineage.json").read_bytes()
    (ROOT / "data/v5_manifest.sha256").write_text(
        f"{hashlib.sha256(manifest_text.encode()).hexdigest()}  data/v5_manifest.json\n"
        f"{hashlib.sha256(lineage_bytes).hexdigest()}  data/dataset_lineage.json\n"
        f"{hashlib.sha256((ROOT / 'data/label_mapping.json').read_bytes()).hexdigest()}  data/label_mapping.json\n")

    cross = Counter()
    for r in included:
        if len(r["cluster_sources"]) > 1 and r["status"] != "excluded_source_semantics":
            cross["+".join(r["cluster_sources"])] += 1
    dedup = {
        "exact_duplicates_collapsed_within_source": dict(exact_dups),
        "status_by_source": {f"{s}/{st}": c for (s, st), c in sorted(status.items()) if s != "cross_source_cluster"},
        "cross_source_clusters": {k: c for (s, k), c in status.items() if s == "cross_source_cluster"},
        "rows_in_cross_source_clusters": dict(cross),
        "tensor_trust_jailbreakllms_overlap_rows": sum(c for k, c in cross.items()
                                                       if "tensor_trust" in k and "jailbreakllms" in k),
        "clusters_spanning_splits": len(leaking),
        "clusters_total": len(set(roots)),
        "method": manifest["dedup"],
    }
    (ROOT / "reports/dataset_dedup_report.json").write_text(json.dumps(dedup, indent=2) + "\n")
    tt_rows = [r for r in rows if r["source"] == "tensor_trust"]
    tt_stats.update(
        included_by_split_label={f"{sp}/{lb}": c for (s, sp, lb), c in sorted(counts.items()) if s == "tensor_trust"},
        included_by_native_category=dict(Counter(r["native_category"] for r in tt_rows
                                                if r.get("status") == "included")),
        status=dict(Counter(r.get("status") for r in tt_rows)),
        security_docs=doc_stats)
    (ROOT / "reports/tensor_trust_training_stats.json").write_text(json.dumps(tt_stats, indent=2) + "\n")
    print(json.dumps({"counts": {f"{s}/{sp}/{lb}": c for (s, sp, lb), c in sorted(counts.items())},
                      "leaking_clusters": len(leaking), "variants": {k: v["n"] for k, v in variant_summary.items()}},
                     indent=1))
    if leaking:
        sys.exit("FAIL: clusters span splits")


if __name__ == "__main__":
    main()
