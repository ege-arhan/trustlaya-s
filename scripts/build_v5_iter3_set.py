"""V5 iteration 3 data: HackAPrompt successful attacks + a fresh, never-scored OOD benign test.

Earlier V5 data (v1 pool, benign-repair rows) is untouched and reused as is.

  hackaprompt  hackaprompt/hackaprompt-dataset @25b87fb (MIT, gated; terms accepted by the
               account owner). Only `correct == True` user inputs become ATTACK (the model
               produced the level's target output). Failed attempts are EXCLUDED, never BENIGN.
  massive_dev  MASSIVE 1.1 *dev* partition, en/de/tr, 1,000 per locale: a fresh OOD benign
               test that no earlier iteration scored. Never TRAIN/DEV.

Clustering (MinHash, Jaccard > 0.70) runs jointly with every earlier row, deepset, Gandalf
and MASSIVE test. HackAPrompt rows that cluster with an OOD row are excluded; rows that
cluster with pool rows take that split; fresh-test rows that cluster with anything else
are dropped. Caps per split are fixed here before training.

Run: .venv/bin/python scripts/build_v5_iter3_set.py
"""

import glob
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
from build_v5_training_set import SEED, _cumulative, cluster, normalize, sha  # noqa: E402
from fetch_tensor_trust import ROOT, load_pins, sha256  # noqa: E402

OUT = ROOT / "data/v5_iter3"
HACKAPROMPT_REV = "25b87fbedfb86840abaf8cd09af7a029208a971a"
CAPS = {"TRAIN": 5000, "DEV": 1000, "TEST": 1000}
FRESH_PER_LOCALE = 1000
OOD = ("deepset", "gandalf", "massive", "massive_dev")


def load_new():
    path = ROOT / "data/external/hackaprompt" / HACKAPROMPT_REV / "hackaprompt.parquet"
    pinned = load_pins()["hackaprompt"]["files"]["hackaprompt.parquet"]
    if sha256(path) != pinned:
        sys.exit("FAIL: hackaprompt.parquet checksum mismatch")
    frame = pd.read_parquet(path)
    expected = {"level", "prompt", "user_input", "completion", "model", "expected_completion", "token_count",
                "correct", "error", "score", "dataset", "timestamp", "session_id"}
    if set(frame.columns) != expected or frame["correct"].dtype != bool:
        sys.exit("FAIL: unexpected HackAPrompt schema")
    stats = {"rows": len(frame), "correct_rows": int(frame["correct"].sum())}
    rows, seen = [], set()
    for item in frame[frame["correct"]].itertuples():
        text = (item.user_input or "").strip()
        key = sha(normalize(text))
        if not text or key in seen:
            continue
        seen.add(key)
        rows.append({"sample_id": f"hackaprompt:{item.Index}", "source": "hackaprompt", "label": "ATTACK",
                     "native_category": f"successful_attack_level{item.level}", "text": text, "language": "en",
                     "provenance": f"HackAPrompt {item.dataset}, model {item.model}, correct=True",
                     "license": "MIT", "contains_pwned": bool(re.search(r"pwn", text, re.I))})
    stats["unique_successful_inputs"] = len(rows)
    stats["failed_attempts_excluded"] = int((~frame["correct"]).sum())
    tarball = ROOT / "data/external/massive/amazon-massive-dataset-1.1.tar.gz"
    if sha256(tarball) != load_pins()["massive"]["files"]["amazon-massive-dataset-1.1.tar.gz"]:
        sys.exit("FAIL: MASSIVE checksum mismatch")
    with tarfile.open(tarball) as tar:
        for locale, lang in (("en-US", "en"), ("de-DE", "de"), ("tr-TR", "tr")):
            items = [json.loads(line) for line in tar.extractfile(f"1.1/data/{locale}.jsonl")]
            items = [x for x in items if x["partition"] == "dev"]
            for x in random.Random(SEED).sample(items, FRESH_PER_LOCALE):
                rows.append({"sample_id": f"massive_dev:{locale}:{x['id']}", "source": "massive_dev",
                             "label": "BENIGN", "native_category": f"assistant_utterance:{x['scenario']}",
                             "text": x["utt"], "language": lang, "license": "CC-BY-4.0",
                             "provenance": "MASSIVE 1.1 dev partition (never scored before iteration 3)"})
    return rows, stats


def main():
    old = [json.loads(line) for line in open(ROOT / "data/v5/all_rows.jsonl")]
    benign = [json.loads(line) for line in open(ROOT / "data/v5_benign/new_rows.jsonl")]
    base = [r for r in old if r["status"] == "included" or r["source"] in ("deepset", "gandalf")] + benign
    new, stats = load_new()
    for r in new:
        r["text_sha256"], r["normalized_sha256"] = sha(r["text"]), sha(normalize(r["text"]))
    combined = base + new
    members = defaultdict(list)
    for i, root in enumerate(cluster(combined)):
        members[root].append(i)
    status = Counter()
    for idx in members.values():
        group = [combined[i] for i in idx]
        pool = [g for g in group if g["source"] not in OOD and g.get("split") not in (None, "OOD_BENIGN_TEST")
                and g["source"] not in ("hackaprompt", "massive_dev")]
        ood = [g for g in group if g["source"] in ("deepset", "gandalf", "massive")]
        key = min(g["normalized_sha256"] for g in group)
        for g in group:
            if g["source"] == "massive_dev":
                g["split"] = "OOD_BENIGN_FRESH"
                g["status"] = "included" if all(x["source"] == "massive_dev" for x in group) else "excluded_overlaps_other"
            elif g["source"] == "hackaprompt":
                splits = {p["split"] for p in pool}
                g["split"] = next(iter(splits)) if len(splits) == 1 else next(
                    name for name, share in _cumulative() if int(key[:8], 16) % 100 < share)
                benign_twin = any(x["label"] == "BENIGN" for x in group)
                g["status"] = ("excluded_overlaps_ood" if ood else
                               "excluded_bridges_splits" if len(splits) > 1 else
                               "excluded_benign_near_duplicate" if benign_twin else "included")
            else:
                continue
            g["cluster"] = key[:16]
            status[(g["source"], g["status"])] += 1
    rng = random.Random(SEED)
    for split, cap in CAPS.items():
        items = [r for r in new if r["source"] == "hackaprompt" and r["status"] == "included" and r["split"] == split]
        if len(items) > cap:
            keep = {id(x) for x in rng.sample(items, cap)}
            for x in items:
                if id(x) not in keep:
                    x["status"] = "excluded_cap"
                    status[("hackaprompt", "excluded_cap")] += 1
                    status[("hackaprompt", "included")] -= 1
    OUT.mkdir(parents=True, exist_ok=True)
    included = [r for r in new if r["status"] == "included"]
    with open(OUT / "new_rows.jsonl", "w") as out:
        for r in included:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"hackaprompt": {**stats, "revision": HACKAPROMPT_REV, "license": "MIT (gated; terms accepted)",
                               "label_rule": "correct == True -> ATTACK; failed attempts excluded"},
               "massive_dev": {"per_locale": FRESH_PER_LOCALE, "partition": "dev", "role": "OOD_BENIGN_FRESH"},
               "caps": CAPS, "status": {f"{s}/{st}": c for (s, st), c in sorted(status.items()) if c},
               "included": {f"{r[0]}/{r[1]}/{r[2]}": c for r, c in sorted(Counter(
                   (r["source"], r["language"], r["split"]) for r in included).items())},
               "hackaprompt_contains_pwned_share": {sp: round(sum(r["contains_pwned"] for r in included if r["source"] == "hackaprompt" and r["split"] == sp)
                                                          / max(1, sum(1 for r in included if r["source"] == "hackaprompt" and r["split"] == sp)), 3)
                                                    for sp in ("TRAIN", "DEV", "TEST")}}
    (ROOT / "reports/iter3_dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["sample_id", "source", "language", "native_category", "split", "status", "cluster", "normalized_sha256"]
    lineage = {"fields": fields, "rows": [[r.get(f) for f in fields] for r in sorted(new, key=lambda r: r["sample_id"])]}
    (ROOT / "data/iter3_lineage.json").write_text(json.dumps(lineage, separators=(",", ":")) + "\n")
    (ROOT / "data/iter3_manifest.sha256").write_text("".join(
        f"{hashlib.sha256((ROOT / p).read_bytes()).hexdigest()}  {p}\n"
        for p in ("data/iter3_lineage.json", "reports/iter3_dataset.json")))
    print(json.dumps({k: summary[k] for k in ("status", "included", "hackaprompt_contains_pwned_share")}, indent=1))


if __name__ == "__main__":
    main()
