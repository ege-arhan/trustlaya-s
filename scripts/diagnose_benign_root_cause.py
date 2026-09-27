"""Root-cause diagnostics for E6 benign false positives (H1-H5). No training.

Uses the frozen E6 checkpoint and its frozen DEV threshold. Data:
- DEV splits of the new benign sources (dolly, oasst2, aya; gold language tags);
- deepset benign rows (already scored once as OOD; used here for analysis only,
  never for selection, thresholds or training). MASSIVE (the new OOD benign
  test) is not touched.
Language for deepset uses a stopword heuristic whose accuracy is measured on
the gold-tagged DEV rows first. Keyword counterfactuals delete the keyword and
rescore; they are perturbation probes, not new data.

Run: .venv/bin/python scripts/diagnose_benign_root_cause.py
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v5_benign_repair_set import KEYWORDS, LENGTH_BUCKETS  # noqa: E402
from fetch_tensor_trust import ROOT  # noqa: E402
from train_v5_ablation import BACKBONE, TOKENIZER  # noqa: E402
from trustlaya.utils import device, normalize  # noqa: E402
from trustlaya.v5_model import V5Classifier, batch  # noqa: E402

E6 = "E6_mixed_balanced_focal_510"
STOP = {
    "en": {"the", "and", "is", "are", "you", "what", "how", "to", "of", "in", "a", "for", "can", "with", "this", "that"},
    "de": {"der", "die", "das", "und", "ist", "ich", "nicht", "wie", "was", "ein", "eine", "zu", "mit", "für", "sie", "du"},
    "tr": {"ve", "bir", "bu", "ne", "için", "mi", "mı", "nasıl", "da", "de", "ile", "çok", "olan", "gibi", "ben", "sen"},
}


def guess_language(text):
    words = re.findall(r"\w+", text.lower())
    scores = {lang: sum(w in stop for w in words) for lang, stop in STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def bucket(n):
    return next(name for name, low, high in LENGTH_BUCKETS if low <= n <= high)


def summarize(scores, threshold):
    if not scores:
        return None
    return {"n": len(scores), "fpr": round(sum(s >= threshold for s in scores) / len(scores), 4),
            "mean_score": round(float(np.mean(scores)), 4), "median_score": round(float(np.median(scores)), 4)}


def main():
    threshold = json.loads((ROOT / f"reports/experiments/{E6}/selection.json").read_text())["threshold"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    model = V5Classifier(BACKBONE, pretrained=False)
    model.load(ROOT / f"models/v5-ablation/{E6}/model.safetensors")
    dev = device()
    model.to(dev).eval()

    def score(texts):
        tokens = [tokenizer(normalize(t), add_special_tokens=False, verbose=False)["input_ids"] for t in texts]
        out = [0.0] * len(texts)
        order = sorted(range(len(texts)), key=lambda i: len(tokens[i]))
        with torch.inference_mode():
            for start in range(0, len(order), 64):
                idx = order[start:start + 64]
                ids, mask = batch([tokens[i] for i in idx], 510, tokenizer.cls_token_id, tokenizer.sep_token_id)
                for i, p in zip(idx, torch.sigmoid(model(ids.to(dev), mask.to(dev))).float().cpu().tolist()):
                    out[i] = p
        return out, [len(t) for t in tokens]

    new = [json.loads(line) for line in open(ROOT / "data/v5_benign/new_rows.jsonl")]
    dev_rows = [r for r in new if r["split"] == "DEV"]
    old = [json.loads(line) for line in open(ROOT / "data/v5/all_rows.jsonl")]
    deepset = [r for r in old if r["source"] == "deepset" and r["status"] == "included" and r["label"] == "BENIGN"]
    old_dev = [r for r in old if r["split"] == "DEV" and r["status"] == "included" and r["label"] == "BENIGN"]

    report = {"model": E6, "threshold": threshold, "note": "E6 frozen; deepset used for analysis only"}
    # Language-ID heuristic accuracy on gold-tagged DEV rows.
    gold = Counter((r["language"], guess_language(r["text"])) for r in dev_rows)
    report["language_id_heuristic_accuracy_on_dev"] = {
        lang: round(gold[(lang, lang)] / max(1, sum(c for (g, _), c in gold.items() if g == lang)), 3)
        for lang in ("en", "de", "tr")}

    dev_scores, dev_len = score([r["text"] for r in dev_rows])
    ds_scores, ds_len = score([r["text"] for r in deepset])
    old_scores, old_len = score([r["text"] for r in old_dev])

    # H1 language and H2 length: new DEV benign by gold language x length bucket.
    cells = defaultdict(list)
    for r, s, n in zip(dev_rows, dev_scores, dev_len):
        cells[(r["language"], bucket(n))].append(s)
        cells[(r["language"], "all")].append(s)
    report["H1_H2_new_dev_benign_language_x_length"] = {f"{l}/{b}": summarize(v, threshold) for (l, b), v in sorted(cells.items())}
    ds_cells = defaultdict(list)
    for r, s, n in zip(deepset, ds_scores, ds_len):
        lang = guess_language(r["text"])
        ds_cells[(lang, bucket(n))].append(s)
        ds_cells[(lang, "all")].append(s)
    report["H1_H2_deepset_benign_heuristic_language_x_length"] = {f"{l}/{b}": summarize(v, threshold) for (l, b), v in sorted(ds_cells.items())}

    # H4 source shift: same language (en) and short bucket, different sources.
    src = defaultdict(list)
    for r, s, n in zip(dev_rows, dev_scores, dev_len):
        if r["language"] == "en" and n <= 94:
            src[r["source"]].append(s)
    for r, s, n in zip(old_dev, old_scores, old_len):
        if n <= 94:
            src[r["source"]].append(s)
    for r, s, n in zip(deepset, ds_scores, ds_len):
        if guess_language(r["text"]) == "en" and n <= 94:
            src["deepset (heuristic en)"].append(s)
    report["H4_source_shift_english_le_94_tokens"] = {k: summarize(v, threshold) for k, v in sorted(src.items())}

    # H3 keywords: FPR by keyword group on all benign diagnostics rows, plus deletion counterfactual.
    pool = [(r["text"], s) for r, s in zip(dev_rows + old_dev, dev_scores + old_scores)]
    kw = {}
    for name, pattern in KEYWORDS.items():
        hits = [(t, s) for t, s in pool if re.search(pattern, t, re.I)]
        entry = {"with_keyword": summarize([s for _, s in hits], threshold)}
        if hits:
            stripped = [re.sub(pattern, " ", t, flags=re.I) for t, _ in hits]
            after, _ = score(stripped)
            entry["keyword_deleted"] = summarize(after, threshold)
        kw[name] = entry
    kw["no_keyword"] = {"with_keyword": summarize([s for t, s in pool if not any(
        re.search(p, t, re.I) for p in KEYWORDS.values())], threshold)}
    report["H3_security_vocabulary"] = kw

    # H5 label semantics: what deepset "benign" rows look like (form only, no relabeling).
    forms = Counter()
    for r in deepset:
        text = r["text"].strip()
        forms["question_mark"] += text.endswith("?")
        forms["imperative_start"] += bool(re.match(r"(?i)^(write|tell|give|show|explain|schreib|erklär|gib|sag|nenn)\b", text))
        forms["attack_cue"] += bool(re.search(r"(?i)\b(ignore|vergiss|ignorier|forget|pretend|you are now|du bist jetzt)\b", text))
    report["H5_deepset_benign_form_counts"] = {"n": len(deepset), **forms}
    (ROOT / "reports/benign_root_cause.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1)[:6000])


if __name__ == "__main__":
    main()
