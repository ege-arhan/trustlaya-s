"""V5 iteration 5: synthetic TRAIN-only augmentation with full provenance. Never used in DEV/TEST.

Sources (all rows carry synthetic=true and provenance fields):
  tr_curated   3nesdeniz/turkish-conversation-prompt-injection @29d7593 (CC BY 4.0), all splits,
               attacks and benign rows; the card states the text is synthetic (author-curated).
  mt_de_attack / mt_tr_attack   OPUS-MT translations of English TRAIN attacks (Tensor Trust,
               JailbreakLLMs, HackAPrompt), seeded sample of rows with <= 96 words.
  mt_de_benign / mt_tr_benign   OPUS-MT translations of short English benign questions from Dolly
               TRAIN (<= 32 tokens, question form): controls for a "translationese = attack" shortcut.

Every synthetic row is clustered (MinHash, Jaccard > 0.70) against all TEST/OOD rows (V5 TEST,
deepset, Gandalf, MASSIVE test/dev, iteration-3 TEST) and dropped on any overlap.
Raw synthetic text stays in git-ignored data/v5_iter5/; tracked files hold hashes and counts.

Run: .venv/bin/python scripts/build_v5_iter5_synthetic.py
"""

import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v5_training_set import SEED, cluster, normalize, sha  # noqa: E402
from fetch_tensor_trust import ROOT, load_pins, sha256  # noqa: E402
from train_v5_iter3 import rows_iter3  # noqa: E402
from trustlaya.utils import device, normalize as model_normalize  # noqa: E402

OUT = ROOT / "data/v5_iter5"
TR_REV = "29d7593984f563c4ad56876aa800b9ffd948a2fb"
MT = {"de": ("Helsinki-NLP/opus-mt-en-de", "6183067f769a302e3861815543b9f312c71b0ca4", "CC-BY-4.0"),
      "tr": ("Helsinki-NLP/opus-mt-tc-big-en-tr", "e539fc16a8a1a0ea5950eb339b595bfcce990e90", "CC-BY-4.0")}
BACK = {"de": ("Helsinki-NLP/opus-mt-de-en", "1a922f3b32a8e809e17a47d4b32142d8105924e5"),
        "tr": ("Helsinki-NLP/opus-mt-tc-big-tr-en", "2261c8fc7b1af59caee87f8ff0ecf3fbccfe8391")}
N_ATTACK = {"de": 1100, "tr": 1000}
N_BENIGN = {"de": 1100, "tr": 1000}
MAX_WORDS = 96
TEST_SPLITS = ("TEST", "OOD_TEST", "OOD_BENIGN_TEST", "OOD_BENIGN_FRESH")
QUESTION = re.compile(r"(?i)(\?\s*$|^(what|how|why|who|when|where|which|can|could|is|are|do|does|would|should)\b)")


def is_question(text):
    return bool(QUESTION.search(text.strip()))


def translate(texts, lang, back=False):
    name, rev = (BACK if back else MT)[lang][:2]
    tok = MarianTokenizer.from_pretrained(name, revision=rev)
    model = MarianMTModel.from_pretrained(name, revision=rev).to(device()).eval()
    out = []
    for start in range(0, len(texts), 16):
        batch = tok(texts[start:start + 16], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device())
        with torch.inference_mode():
            gen = model.generate(**batch, num_beams=4, max_new_tokens=320)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def synthetic_row(source, sid, text, label, lang, method, original, model_name, original_text=None):
    return {"sample_id": f"{source}:{sid}", "source": source, "label": label, "text": text, "language": lang,
            "split": "TRAIN", "status": "included", "synthetic": True, "synthetic_method": method,
            "original_source": original.get("source"), "original_example_id": original.get("sample_id"),
            "source_language": original.get("language", "en"), "target_language": lang,
            "generation_or_translation_model": model_name,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_checksum": original.get("checksum"), "question_form": is_question(text),
            "original_text_sha256": sha(original_text) if original_text else None}


def main():
    rng = random.Random(SEED)
    rows = rows_iter3()
    base = OUT / "sources"
    tr_dir = ROOT / "data/external/tr_conversation_pi" / TR_REV
    pins = load_pins()["tr_conversation_pi"]["files"]
    synthetic = []
    for split in ("train", "validation", "test"):
        path = tr_dir / f"data/{split}.jsonl"
        if sha256(path) != pins[f"data/{split}.jsonl"]:
            sys.exit("FAIL: Turkish dataset checksum mismatch")
        for line in path.open():
            item = json.loads(line)
            if item["label"] not in (0, 1) or item["source_type"] != "synthetic_curated":
                sys.exit("FAIL: unexpected Turkish dataset schema")
            synthetic.append(synthetic_row(
                "tr_curated", item["id"], item["text"], "ATTACK" if item["label"] else "BENIGN", "tr",
                "author_curated (card: text is synthetic)",
                {"source": "3nesdeniz/turkish-conversation-prompt-injection", "sample_id": item["id"], "language": "tr",
                 "checksum": pins[f"data/{split}.jsonl"]}, "none (human-curated synthetic)"))
            synthetic[-1]["native_category"] = item["category"]
    # English originals: TRAIN attacks and short benign questions (no DEV/TEST rows are touched).
    attacks = [r for r in rows if r["status"] == "included" and r["split"] == "TRAIN" and r["label"] == "ATTACK"
               and r["source"] in ("tensor_trust", "jailbreakllms", "hackaprompt") and len(r["text"].split()) <= MAX_WORDS]
    per_source = defaultdict(list)
    for r in attacks:
        per_source[r["source"]].append(r)
    originals = []
    share = N_ATTACK["de"] // len(per_source)
    for source in sorted(per_source):
        originals += rng.sample(per_source[source], min(share, len(per_source[source])))
    originals = originals[:N_ATTACK["de"]]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v2")
    dolly = [r for r in rows if r["source"] == "dolly" and r["split"] == "TRAIN" and r["status"] == "included"
             and is_question(r["text"]) and len(tokenizer(model_normalize(r["text"]), add_special_tokens=False)["input_ids"]) <= 32]
    benign_originals = rng.sample(dolly, N_BENIGN["de"])
    checksum = {"tensor_trust": (ROOT / "data/v5_manifest.sha256").read_text().split()[0],
                "jailbreakllms": (ROOT / "data/v5_manifest.sha256").read_text().split()[0],
                "hackaprompt": load_pins()["hackaprompt"]["files"]["hackaprompt.parquet"],
                "dolly": load_pins()["dolly"]["files"]["databricks-dolly-15k.jsonl"]}
    started = time.time()
    for lang in ("de", "tr"):
        model_name = f"{MT[lang][0]}@{MT[lang][1][:10]} ({MT[lang][2]})"
        chosen = originals[:N_ATTACK[lang]]
        for original, text in zip(chosen, translate([r["text"] for r in chosen], lang)):
            synthetic.append(synthetic_row(f"mt_{lang}_attack", original["sample_id"].replace(":", "_"), text, "ATTACK", lang,
                                           "machine_translation", {**original, "checksum": checksum[original["source"]]},
                                           model_name, original["text"]))
        chosen = benign_originals[:N_BENIGN[lang]]
        for original, text in zip(chosen, translate([r["text"] for r in chosen], lang)):
            synthetic.append(synthetic_row(f"mt_{lang}_benign", original["sample_id"].replace(":", "_"), text, "BENIGN", lang,
                                           "machine_translation", {**original, "checksum": checksum["dolly"]},
                                           model_name, original["text"]))
    translation_seconds = round(time.time() - started, 1)
    # Leakage: drop any synthetic row that clusters with a TEST/OOD row.
    for r in synthetic:
        r["text_sha256"], r["normalized_sha256"] = sha(r["text"]), sha(normalize(r["text"]))
    tests = [r for r in rows if r.get("split") in TEST_SPLITS and r.get("status") == "included"]
    combined = tests + synthetic
    members = defaultdict(list)
    for i, root in enumerate(cluster(combined)):
        members[root].append(i)
    dropped = Counter()
    for idx in members.values():
        group = [combined[i] for i in idx]
        if any(not g.get("synthetic") for g in group):
            for g in group:
                if g.get("synthetic"):
                    g["status"] = "excluded_overlaps_test"
                    dropped[(g["source"], ",".join(sorted({x["source"] for x in group if not x.get("synthetic")})))] += 1
    kept = [r for r in synthetic if r["status"] == "included"]
    for r in kept:
        r["tokens_count"] = len(tokenizer(model_normalize(r["text"]), add_special_tokens=False)["input_ids"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "synthetic_rows.jsonl", "w") as out:
        for r in kept:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Back-translation sample for the audit (semantic preservation).
    audit = {}
    for lang in ("de", "tr"):
        sample = [r for r in kept if r["source"] == f"mt_{lang}_attack"][:300]
        back = translate([r["text"] for r in sample], lang, back=True)
        originals_by_id = {r["sample_id"]: r["text"] for r in attacks}
        audit[lang] = [{"original": originals_by_id.get(r["original_example_id"], ""), "translation": r["text"],
                        "back": b} for r, b in zip(sample, back)]
    with open(OUT / "backtranslation_sample.json", "w") as out:
        json.dump(audit, out, ensure_ascii=False)
    fields = ["sample_id", "source", "label", "language", "synthetic", "synthetic_method", "original_source",
              "original_example_id", "source_language", "target_language", "generation_or_translation_model",
              "generation_timestamp", "source_checksum", "question_form", "text_sha256", "original_text_sha256",
              "status", "tokens_count"]
    manifest = {
        "schema": "v5-iter5-synthetic-1", "role": "TRAIN AUGMENTATION ONLY (not gold, not human-labeled, not a test)",
        "sources": {"tr_curated": {"repo": "3nesdeniz/turkish-conversation-prompt-injection", "revision": TR_REV,
                                   "license": "CC-BY-4.0", "card": "text is synthetic (author-curated)"},
                    "machine_translation": {lang: {"model": MT[lang][0], "revision": MT[lang][1], "license": MT[lang][2]}
                                            for lang in MT},
                    "backtranslation_for_audit_only": {lang: {"model": BACK[lang][0], "revision": BACK[lang][1]} for lang in BACK}},
        "sampling": {"attack_originals": f"TRAIN attacks <= {MAX_WORDS} words, equal per source, seed {SEED}",
                     "benign_originals": "Dolly TRAIN, question form, <= 32 tokens, seed 42", "counts": {"attack": N_ATTACK, "benign": N_BENIGN}},
        "translation_seconds": translation_seconds,
        "dropped_overlapping_test": {f"{s}|{t}": c for (s, t), c in sorted(dropped.items())},
        "kept_by_source_label": dict(sorted(Counter(f"{r['source']}/{r['label']}" for r in kept).items())),
        "rows": {"fields": fields, "rows": [[r.get(f) for f in fields] for r in kept]},
    }
    text = json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"
    (ROOT / "data/v5_iter5_manifest.json").write_text(text)
    (ROOT / "data/v5_iter5_manifest.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "  data/v5_iter5_manifest.json\n")
    print(json.dumps({k: manifest[k] for k in ("kept_by_source_label", "dropped_overlapping_test", "translation_seconds")}, indent=1))


if __name__ == "__main__":
    main()
