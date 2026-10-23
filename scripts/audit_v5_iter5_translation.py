"""Translation-artifact audit for the iteration-5 synthetic data (no text is written to reports).

Writes reports/v5_iter5_translation_audit.md and reports/v5_iter5_short_attack_analysis.json.
chrF (character 6-gram F-score, beta 2) compares each English original with its back-translation.

Run: .venv/bin/python scripts/audit_v5_iter5_translation.py
"""

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v5_iter5_synthetic import is_question  # noqa: E402
from fetch_tensor_trust import ROOT  # noqa: E402
from train_v5_iter3 import rows_iter3  # noqa: E402
from trustlaya.utils import normalize  # noqa: E402

BUCKETS = (("<=32", 0, 32), ("33-64", 33, 64), ("65-94", 65, 94), ("95-256", 95, 256), ("257+", 257, 10 ** 9))


def chrf(ref, hyp, n=6, beta=2.0):
    scores = []
    for k in range(1, n + 1):
        r = Counter(ref[i:i + k] for i in range(len(ref) - k + 1))
        h = Counter(hyp[i:i + k] for i in range(len(hyp) - k + 1))
        if not r or not h:
            continue
        overlap = sum((r & h).values())
        p, q = overlap / sum(h.values()), overlap / sum(r.values())
        scores.append(0.0 if p + q == 0 else (1 + beta ** 2) * p * q / (beta ** 2 * p + q))
    return sum(scores) / len(scores) if scores else 0.0


def bucket(n):
    return next(name for name, lo, hi in BUCKETS if lo <= n <= hi)


def main():
    tok = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v2")
    count = lambda t: len(tok(normalize(t), add_special_tokens=False, verbose=False)["input_ids"])
    synth = [json.loads(line) for line in open(ROOT / "data/v5_iter5/synthetic_rows.jsonl")]
    rows = {r["sample_id"]: r for r in rows_iter3()}
    groups = defaultdict(list)
    for r in synth:
        groups[r["source"] + ("_" + r["label"].lower() if r["source"] == "tr_curated" else "")].append(r)
    real_ref = {"real_de_benign (OASST2/Aya)": [r for r in rows.values() if r.get("language") == "de" and r["source"] in ("oasst2", "aya") and r["split"] == "TRAIN"],
                "real_tr_benign (Aya/OASST2)": [r for r in rows.values() if r.get("language") == "tr" and r["source"] in ("oasst2", "aya") and r["split"] == "TRAIN"],
                "english_attack_originals": [rows[r["original_example_id"]] for r in synth if r["source"] == "mt_de_attack" and r["original_example_id"] in rows]}
    stats = {}
    for name, items in list(groups.items()) + list(real_ref.items()):
        texts = [r["text"] for r in items]
        tokens = [count(t) for t in texts]
        words = [max(1, len(t.split())) for t in texts]
        stats[name] = {"n": len(texts), "median_chars": statistics.median(len(t) for t in texts),
                       "median_words": statistics.median(words), "median_tokens": statistics.median(tokens),
                       "tokens_per_word": round(sum(tokens) / sum(words), 3),
                       "question_share": round(sum(is_question(t) for t in texts) / len(texts), 3),
                       "exclamation_share": round(sum("!" in t for t in texts) / len(texts), 3),
                       "quote_share": round(sum(bool(re.search(r"[\"'„“”«»]", t)) for t in texts) / len(texts), 3),
                       "length_buckets": dict(Counter(bucket(n) for n in tokens)),
                       "top_word_trigrams": [" ".join(k) for k, _ in Counter(
                           tuple(w) for t in texts for w in zip(*(t.lower().split()[i:] for i in range(3)))).most_common(5)]
                       if name.startswith("mt_") else None}
    # Same English original in en / de / tr.
    by_original = defaultdict(dict)
    for r in synth:
        if r["source"] in ("mt_de_attack", "mt_tr_attack") and r["original_example_id"] in rows:
            by_original[r["original_example_id"]][r["target_language"]] = r["text"]
    triples = [(rows[k]["text"], v["de"], v["tr"]) for k, v in by_original.items() if "de" in v and "tr" in v]
    q_en = [is_question(e) for e, _, _ in triples]
    parallel = {"n": len(triples),
                "question_kept_de": round(sum(is_question(d) for (e, d, _), q in zip(triples, q_en) if q) / max(1, sum(q_en)), 3),
                "question_kept_tr": round(sum(is_question(t) for (e, _, t), q in zip(triples, q_en) if q) / max(1, sum(q_en)), 3),
                "median_char_ratio_de_en": round(statistics.median(len(d) / max(1, len(e)) for e, d, _ in triples), 3),
                "median_char_ratio_tr_en": round(statistics.median(len(t) / max(1, len(e)) for e, _, t in triples), 3),
                "median_token_ratio_de_en": round(statistics.median(count(d) / max(1, count(e)) for e, d, _ in triples), 3),
                "median_token_ratio_tr_en": round(statistics.median(count(t) / max(1, count(e)) for e, _, t in triples), 3),
                "placeholder_or_code_kept": {lang: round(sum(("{" in e) <= ("{" in x) for e, x in pairs) / max(1, len(pairs)), 3)
                                             for lang, pairs in (("de", [(e, d) for e, d, _ in triples]), ("tr", [(e, t) for e, _, t in triples]))}}
    back = json.loads((ROOT / "data/v5_iter5/backtranslation_sample.json").read_text())
    semantic = {}
    for lang, items in back.items():
        scores = [chrf(i["original"].lower(), i["back"].lower()) for i in items if i["original"]]
        semantic[lang] = {"n": len(scores), "chrf_median": round(statistics.median(scores), 3),
                          "chrf_p10": round(sorted(scores)[len(scores) // 10], 3), "share_below_0_4": round(sum(s < 0.4 for s in scores) / len(scores), 3)}
    short = {name: {"n": s["n"], "length_buckets": s["length_buckets"], "question_share": s["question_share"]}
             for name, s in stats.items()}
    (ROOT / "reports/v5_iter5_short_attack_analysis.json").write_text(json.dumps(
        {"note": "Synthetic TRAIN augmentation only; not a test set.", "by_source": short}, indent=2) + "\n")
    rows_md = "\n".join(f"| {n} | {s['n']} | {s['median_chars']} | {s['median_tokens']} | {s['tokens_per_word']} | {s['question_share']} | {s['exclamation_share']} | {s['quote_share']} |"
                        for n, s in stats.items())
    md = f"""# V5 iteration 5: translation audit (synthetic TRAIN augmentation)

All numbers describe synthetic TRAIN data. None of it is gold, human-labeled or used for testing.
Tokens use the V2 tokenizer. Full numbers: `reports/v5_iter5_short_attack_analysis.json` and below.

| Group | n | median chars | median tokens | tokens/word | question share | "!" share | quote share |
|---|---|---|---|---|---|---|---|
{rows_md}

## Same English attack in English / German / Turkish ({parallel['n']} originals)

- Question form kept when the English was a question: German {parallel['question_kept_de']}, Turkish {parallel['question_kept_tr']}.
- Median length ratio vs English: characters de {parallel['median_char_ratio_de_en']}, tr {parallel['median_char_ratio_tr_en']};
  tokens de {parallel['median_token_ratio_de_en']}, tr {parallel['median_token_ratio_tr_en']}.
- Braces/placeholders kept: de {parallel['placeholder_or_code_kept']['de']}, tr {parallel['placeholder_or_code_kept']['tr']}.

## Semantic preservation (back-translation to English, chrF, 300 attacks per language)

| Language | chrF median | chrF 10th percentile | share below 0.40 |
|---|---|---|---|
| de | {semantic['de']['chrf_median']} | {semantic['de']['chrf_p10']} | {semantic['de']['share_below_0_4']} |
| tr | {semantic['tr']['chrf_median']} | {semantic['tr']['chrf_p10']} | {semantic['tr']['share_below_0_4']} |

chrF compares surface characters, so it underestimates meaning preservation for paraphrases and
overestimates it when the model copies English words. It is a screening number, not a quality
judgment. Frequent translated trigrams per source are listed in the JSON sidecar below.

```json
{json.dumps({n: s['top_word_trigrams'] for n, s in stats.items() if s['top_word_trigrams']}, ensure_ascii=False, indent=1)}
```
"""
    (ROOT / "reports/v5_iter5_translation_audit.md").write_text(md)
    print(json.dumps({"parallel": parallel, "semantic": semantic}, indent=1))


if __name__ == "__main__":
    main()
