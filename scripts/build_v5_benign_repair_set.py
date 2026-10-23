"""Benign-repair dataset for V5 iteration 2. The V5 v1 data stays untouched.

Adds public, source-labeled benign user text in English, German and Turkish
to the frozen V5 pool (data/v5/all_rows.jsonl, same splits/clusters):

  dolly   databricks-dolly-15k instructions (en, CC BY-SA 3.0)
  oasst2  OpenAssistant oasst2 first user prompts (en/de/tr, Apache-2.0)
  aya     Aya dataset human-written prompts (en/de/tr, Apache-2.0)
  massive MASSIVE 1.1 test utterances (en-US/de-DE/tr-TR, CC BY 4.0)
          -> TEST2_OOD_BENIGN only, never TRAIN/DEV

deepset stays OOD test only. New rows are clustered (MinHash, Jaccard > 0.70)
together with every existing row: a new row that clusters with a pool row takes
that row's split; with an OOD row (deepset/Gandalf/MASSIVE) it leaves the pool;
a benign near-duplicate of an attack is excluded. TRAIN/DEV/TEST caps per
(source, language) are fixed below before any training.

Run: .venv/bin/python scripts/build_v5_benign_repair_set.py [--record]
"""

import argparse
import glob
import gzip
import hashlib
import json
import random
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v5_training_set import EXTERNAL, HF, JACCARD, SEED, _cumulative, cluster, normalize, sha  # noqa: E402
from fetch_tensor_trust import ROOT, fetch_pinned  # noqa: E402

OUT = ROOT / "data/v5_benign"
CAPS = {"TRAIN": 3000, "DEV": 600, "TEST": 600}  # per (source, language)
MASSIVE_PER_LOCALE = 1000
NEW_POOL = ("dolly", "oasst2", "aya")
OOD = ("deepset", "gandalf", "massive")
KEYWORDS = {
    "ignore": r"\bignor\w*", "bypass": r"\bbypass\w*", "override": r"\boverrid\w*", "system": r"\bsystem\w*",
    "admin": r"\badmin\w*", "developer": r"\bdeveloper\w*", "instruction": r"\binstruction\w*",
    "prompt": r"\bprompt\w*", "jailbreak": r"\bjailbreak\w*", "security": r"\bsecurit\w*",
    "password": r"\bpassw\w*|\bparola\w*|\bşifre\w*|\bpasswort\w*", "token": r"\btoken\w*",
    "api_key": r"\bapi[ _-]?key\w*",
}
LENGTH_BUCKETS = (("<=32", 0, 32), ("33-64", 33, 64), ("65-94", 65, 94), ("95-256", 95, 256), ("257+", 257, 10 ** 9))
LANG = {"English": "en", "German": "de", "Turkish": "tr", "en": "en", "de": "de", "tr": "tr",
        "en-US": "en", "de-DE": "de", "tr-TR": "tr"}


def new_row(source, native_id, text, language, native, provenance, license):
    return {"sample_id": f"{source}:{native_id}", "source": source, "native_category": native,
            "label": "BENIGN", "text": text, "language": language, "provenance": provenance,
            "license": license}


def load_sources(record):
    rows = []
    rev = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
    path = fetch_pinned("dolly", "https://huggingface.co/datasets/databricks/databricks-dolly-15k", rev,
                        {p: HF.format(repo="databricks/databricks-dolly-15k", rev=rev, path=p)
                         for p in ("databricks-dolly-15k.jsonl", "README.md")},
                        EXTERNAL / "dolly" / rev, record=record)["databricks-dolly-15k.jsonl"]
    for i, line in enumerate(path.open()):
        item = json.loads(line)
        rows.append(new_row("dolly", i, item["instruction"], "en", f"instruction:{item['category']}",
                            "Databricks employees wrote instructions (human)", "CC-BY-SA-3.0"))
    rev = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
    name = "2023-11-05_oasst2_ready.messages.jsonl.gz"
    path = fetch_pinned("oasst2", "https://huggingface.co/datasets/OpenAssistant/oasst2", rev,
                        {p: HF.format(repo="OpenAssistant/oasst2", rev=rev, path=p) for p in (name, "README.md")},
                        EXTERNAL / "oasst2" / rev, record=record)[name]
    for line in gzip.open(path, "rt"):
        m = json.loads(line)
        if (m["role"] == "prompter" and m["parent_id"] is None and m["lang"] in ("en", "de", "tr")
                and m["review_result"] and not m["deleted"] and not m["synthetic"]
                and (m.get("labels") or {}).get("spam", {}).get("value", 0) < 0.5):
            rows.append(new_row("oasst2", m["message_id"], m["text"], m["lang"], "first_user_prompt",
                                "volunteer-written first user turn, passed OASST review, not spam-labeled", "Apache-2.0"))
    rev = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
    files = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]
    paths = fetch_pinned("aya", "https://huggingface.co/datasets/CohereLabs/aya_dataset", rev,
                         {p: HF.format(repo="CohereLabs/aya_dataset", rev=rev, path=p) for p in files + ["README.md"]},
                         EXTERNAL / "aya" / rev, record=record)
    for key in files:
        frame = pd.read_parquet(paths[key])
        frame = frame[frame["language"].isin(["English", "German", "Turkish"])]
        for index, item in frame.iterrows():
            rows.append(new_row("aya", f"{key.split('/')[1].split('-')[0]}:{index}", item["inputs"],
                                LANG[item["language"]], f"prompt:{item['annotation_type']}",
                                "Aya annotators wrote or re-annotated the prompt (human)", "Apache-2.0"))
    tarball = fetch_pinned("massive", "https://github.com/alexa/massive (S3 release 1.1)",
                           "1.1-etag-51e0da2a3ff7a016f109e1d1b4306e93-3",
                           {"amazon-massive-dataset-1.1.tar.gz":
                            "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"},
                           EXTERNAL / "massive", record=record)["amazon-massive-dataset-1.1.tar.gz"]
    with tarfile.open(tarball) as tar:
        for locale in ("en-US", "de-DE", "tr-TR"):
            items = [json.loads(line) for line in tar.extractfile(f"1.1/data/{locale}.jsonl")]
            items = [x for x in items if x["partition"] == "test"]
            for x in random.Random(SEED).sample(items, MASSIVE_PER_LOCALE):
                rows.append(new_row("massive", f"{locale}:{x['id']}", x["utt"], LANG[locale],
                                    f"assistant_utterance:{x['scenario']}",
                                    "crowd-written (en) / professionally localized (de, tr) voice-assistant request",
                                    "CC-BY-4.0"))
    return rows


def keyword_groups(text):
    return [name for name, pattern in KEYWORDS.items() if re.search(pattern, text, re.I)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    old = [json.loads(line) for line in open(ROOT / "data/v5/all_rows.jsonl")]
    new = load_sources(args.record)
    seen, dups = set(), Counter()
    unique = []
    for r in new:
        r["text"] = r["text"].strip()
        r["text_sha256"] = sha(r["text"])
        r["normalized_sha256"] = sha(normalize(r["text"]))
        key = (r["source"], r["normalized_sha256"])
        if not r["text"] or key in seen:
            dups[r["source"]] += 1
            continue
        seen.add(key)
        unique.append(r)
    new = unique
    base = [r for r in old if r["status"] == "included" or r["source"] in ("deepset", "gandalf")]
    combined = base + new
    roots = cluster(combined)
    members = defaultdict(list)
    for i, root in enumerate(roots):
        members[root].append(i)
    status = Counter()
    for idx in members.values():
        group = [combined[i] for i in idx]
        old_pool = [g for g in group if g["source"] not in NEW_POOL + OOD]
        ood = [g for g in group if g["source"] in OOD]
        attack = any(g["label"] == "ATTACK" for g in group)
        key = min(g["normalized_sha256"] for g in group)
        split = old_pool[0]["split"] if old_pool else next(
            name for name, share in _cumulative() if int(key[:8], 16) % 100 < share)
        for g in group:
            if g["source"] not in NEW_POOL + ("massive",):
                continue
            g["cluster"] = key[:16]
            if g["source"] == "massive":
                g["split"] = "OOD_BENIGN_TEST"
                g["status"] = "excluded_overlaps_pool" if len(group) > sum(x["source"] == "massive" for x in group) else "included"
            else:
                g["split"] = split
                g["status"] = ("excluded_near_duplicate_of_attack" if attack else
                               "excluded_overlaps_ood" if ood else
                               "excluded_bridges_splits" if len({o["split"] for o in old_pool}) > 1 else "included")
            status[(g["source"], g["status"])] += 1

    # Declared caps per (source, language, split); seed-sampled.
    rng = random.Random(SEED)
    by_key = defaultdict(list)
    for r in new:
        if r["status"] == "included" and r["source"] in NEW_POOL:
            by_key[(r["source"], r["language"], r["split"])].append(r)
    for (source, language, split), items in sorted(by_key.items()):
        if len(items) > CAPS[split]:
            keep = set(id(x) for x in rng.sample(items, CAPS[split]))
            for x in items:
                if id(x) not in keep:
                    x["status"] = "excluded_cap"
                    status[(source, "excluded_cap")] += 1
                    status[(source, "included")] -= 1

    from transformers import AutoTokenizer
    from trustlaya.utils import normalize as model_normalize
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v2")
    for r in new:
        r["tokens_count"] = len(tokenizer(model_normalize(r["text"]), add_special_tokens=False, verbose=False)["input_ids"])
        r["keywords"] = keyword_groups(r["text"])
    OUT.mkdir(parents=True, exist_ok=True)
    included = [r for r in new if r["status"] == "included"]
    with open(OUT / "new_rows.jsonl", "w") as out:
        for r in included:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    def bucket(n):
        return next(name for name, low, high in LENGTH_BUCKETS if low <= n <= high)

    summary = {
        "sources": {
            "dolly": {"repo": "databricks/databricks-dolly-15k", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
                      "license": "CC-BY-SA-3.0", "field": "instruction", "languages": ["en"]},
            "oasst2": {"repo": "OpenAssistant/oasst2", "revision": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
                       "license": "Apache-2.0", "field": "first prompter message", "languages": ["en", "de", "tr"],
                       "filter": "review_result, not deleted, not synthetic, spam label < 0.5"},
            "aya": {"repo": "CohereLabs/aya_dataset", "revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
                    "license": "Apache-2.0", "field": "inputs", "languages": ["en", "de", "tr"]},
            "massive": {"repo": "alexa/massive 1.1 (S3 tarball, ETag 51e0da2a3ff7a016f109e1d1b4306e93-3)",
                        "license": "CC-BY-4.0", "field": "utt (test partition)", "languages": ["en", "de", "tr"],
                        "role": "TEST2_OOD_BENIGN only", "per_locale": MASSIVE_PER_LOCALE},
        },
        "caps_per_source_language": CAPS,
        "exact_duplicates_dropped": dict(dups),
        "status": {f"{s}/{st}": c for (s, st), c in sorted(status.items()) if c},
        "included_counts": {f"{r[0]}/{r[1]}/{r[2]}": c for r, c in sorted(Counter(
            (r["source"], r["language"], r["split"]) for r in included).items())},
        "length_buckets": {f"{s}/{sp}": dict(Counter(bucket(r["tokens_count"]) for r in included
                                                   if r["source"] == s and r["split"] == sp))
                           for s in NEW_POOL + ("massive",) for sp in ("TRAIN", "DEV", "TEST", "OOD_BENIGN_TEST")
                           if any(r["source"] == s and r["split"] == sp for r in included)},
        "security_vocabulary_rows": {f"{s}/{sp}": dict(Counter(k for r in included if r["source"] == s and r["split"] == sp
                                                                for k in r["keywords"]))
                                     for s in NEW_POOL + ("massive",) for sp in ("TRAIN", "DEV", "TEST", "OOD_BENIGN_TEST")
                                     if any(r["source"] == s and r["split"] == sp for r in included)},
        "short_instruction_rows_le_94_tokens": sum(r["tokens_count"] <= 94 for r in included if r["split"] == "TRAIN"),
        "hard_negative_rows_with_security_vocabulary_TRAIN": sum(bool(r["keywords"]) for r in included if r["split"] == "TRAIN"),
    }
    (ROOT / "reports/benign_repair_dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    lineage_fields = ["sample_id", "source", "language", "native_category", "split", "status", "cluster",
                      "normalized_sha256", "tokens_count"]
    lineage = {"fields": lineage_fields, "rows": [[r.get(f) for f in lineage_fields]
                                                 for r in sorted(new, key=lambda r: r["sample_id"])]}
    (ROOT / "data/benign_repair_lineage.json").write_text(json.dumps(lineage, separators=(",", ":")) + "\n")
    (ROOT / "data/benign_repair_manifest.sha256").write_text("".join(
        f"{hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}  {p}\n"
        for p in ("data/benign_repair_lineage.json", "reports/benign_repair_dataset.json")))
    print(json.dumps({k: summary[k] for k in ("status", "included_counts", "short_instruction_rows_le_94_tokens",
                                              "hard_negative_rows_with_security_vocabulary_TRAIN")}, indent=1))


if __name__ == "__main__":
    main()
