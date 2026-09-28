"""V5 iteration 4: threshold strategies T1-T4 on existing checkpoints (DEV selects, TEST reports).

No training. Models: iteration-3 H1 (A4 data + HackAPrompt) and the A4 control, seeds 42-44.
Strategy choice uses DEV only, with a criterion fixed before scoring:
  maximize the worst DEV stratum score, where strata are
  attack recall for length buckets <=32, 33-64, 65-94, 95-256, 257+ and for question-form attacks,
  and benign accuracy (1 - FPR) for each language en / de / tr.
TEST sets (deepset, Gandalf, MASSIVE test/dev, TEST1) were scored in earlier iterations and are
reused for reporting only. deepset language is a stopword guess (marked *).

Strategies:
  T1 global threshold, class-balanced DEV accuracy (the iteration-3 rule)
  T2 T1 subject to DEV recall >= 0.95 on attacks of <= 94 tokens
  T3 global threshold, objective = mean(attack-group accuracy, language-balanced benign accuracy)
  T4 DEV temperature scaling, then threshold 0.5 on calibrated probability

Run: .venv/bin/python scripts/threshold_ablation_v5_iter4.py
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_benign_root_cause import guess_language  # noqa: E402
from evaluate_v5_evidence import fit_temperature, metrics, scale  # noqa: E402
from fetch_tensor_trust import ROOT  # noqa: E402
from train_v5_ablation import BACKBONE, MIXED, TOKENIZER, load_split, predict, tokenize  # noqa: E402
from train_v5_benign_repair import CONTEXT, NEW_SOURCES  # noqa: E402
from train_v5_iter3 import DEV_SOURCES, rows_iter3  # noqa: E402
from trustlaya.utils import device  # noqa: E402
from trustlaya.v5_model import V5Classifier  # noqa: E402

MODELS = {"H1_s42": "models/v5-iter3/H1_grouped_s42", "H1_s43": "models/v5-iter3/H1_grouped_s43",
          "H1_s44": "models/v5-iter3/H1_grouped_s44", "A4_s42": "models/v5-benign-repair/A4_multilingual_4x",
          "A4_s43": "models/v5-variance/A4data_grouped_s43", "A4_s44": "models/v5-variance/A4data_grouped_s44"}
GRID = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]
BUCKETS = (("<=32", 0, 32), ("33-64", 33, 64), ("65-94", 65, 94), ("95-256", 95, 256), ("257+", 257, 10 ** 9))


def lang(r):
    return r["language"] if r.get("language") in ("en", "de", "tr") else (
        guess_language(r["text"]) + "*" if r["source"] == "deepset" else "en?")


def is_question(text):
    return bool(re.search(r"\?\s*$", text.strip())) or bool(re.match(
        r"(?i)^(what|how|why|who|when|where|can|could|is|are|do|does|was|wie|warum|kannst|ist|nedir|nasıl|neden|ne)\b", text.strip()))


def bucket(n):
    return next(name for name, lo, hi in BUCKETS if lo <= n <= hi)


def strata(rows):
    s = defaultdict(list)
    for i, r in enumerate(rows):
        if r["y"] == 1:
            s[f"attack_len_{bucket(len(r['tokens']))}"].append(i)
            if is_question(r["text"]):
                s["attack_question"].append(i)
        else:
            s[f"benign_lang_{lang(r)}"].append(i)
    return s


def stratum_scores(rows, probs, threshold, strat):
    out = {}
    for name, idx in strat.items():
        hits = [probs[i] >= threshold for i in idx]
        out[name] = round(float(np.mean(hits) if name.startswith("attack") else 1 - np.mean(hits)), 4)
    return out


def worst(scores):
    keys = [k for k in scores if k.startswith("attack") or k in ("benign_lang_en", "benign_lang_de", "benign_lang_tr")]
    return min(scores[k] for k in keys)


def thresholds(rows, probs):
    y = np.array([r["y"] for r in rows]); p = np.asarray(probs)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["source"], r["y"])].append(i)
    def class_balanced(t):
        acc = {g: np.mean((p[idx] >= t) == bool(g[1])) for g, idx in groups.items()}
        return (np.mean([v for g, v in acc.items() if g[1]]) + np.mean([v for g, v in acc.items() if not g[1]])) / 2
    short = [i for i, r in enumerate(rows) if r["y"] == 1 and len(r["tokens"]) <= 94]
    langs = defaultdict(list)
    for i, r in enumerate(rows):
        if r["y"] == 0:
            langs[lang(r)].append(i)
    def language_balanced(t):
        attack = np.mean([np.mean(p[idx] >= t) for g, idx in groups.items() if g[1]])
        benign = np.mean([np.mean(p[idx] < t) for l, idx in langs.items() if l in ("en", "de", "tr")])
        return (attack + benign) / 2
    t1 = max(GRID, key=lambda t: (class_balanced(t), -abs(t - 0.5)))
    feasible = [t for t in GRID if np.mean(p[short] >= t) >= 0.95]
    t2 = max(feasible, key=lambda t: (class_balanced(t), -abs(t - 0.5))) if feasible else min(GRID)
    t3 = max(GRID, key=lambda t: (language_balanced(t), -abs(t - 0.5)))
    temperature = fit_temperature(y, p)
    t4 = float(scale(np.asarray([0.5]), 1 / temperature)[0])  # raw threshold equivalent to 0.5 calibrated
    return {"T1": t1, "T2": t2, "T3": t3, "T4": round(t4, 4)}, temperature


def main():
    rows = rows_iter3()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    sets = {"DEV": load_split(rows, "DEV", DEV_SOURCES),
            "TEST1": load_split(rows, "TEST", set(MIXED) | set(NEW_SOURCES) | {"hackaprompt"}),
            "TEST2": load_split(rows, "OOD_TEST", {"deepset", "gandalf"}),
            "MASSIVE_TEST": load_split(rows, "OOD_BENIGN_TEST", {"massive"}),
            "MASSIVE_DEV_FRESH": load_split(rows, "OOD_BENIGN_FRESH", {"massive_dev"})}
    for subset in sets.values():
        tokenize(tokenizer, subset)
    dev_strata = strata(sets["DEV"])
    report = {"doc": __doc__.strip(), "dev_strata_sizes": {k: len(v) for k, v in dev_strata.items()}, "models": {}}
    dev = device()
    for name, folder in MODELS.items():
        model = V5Classifier(BACKBONE, pretrained=False)
        model.load(ROOT / folder / "model.safetensors")
        model.to(dev)
        probs = {k: predict(model, s, CONTEXT, tokenizer, dev)[0] for k, s in sets.items()}
        chosen, temperature = thresholds(sets["DEV"], probs["DEV"])
        entry = {"thresholds": chosen, "temperature": round(temperature, 4), "strategies": {}}
        for strategy, t in chosen.items():
            dev_scores = stratum_scores(sets["DEV"], probs["DEV"], t, dev_strata)
            test = {}
            for split in ("TEST1", "TEST2", "MASSIVE_TEST", "MASSIVE_DEV_FRESH"):
                rs, ps = sets[split], np.asarray(probs[split])
                for source in sorted({r["source"] for r in rs}):
                    idx = [i for i, r in enumerate(rs) if r["source"] == source]
                    m = metrics([rs[i]["y"] for i in idx], ps[idx], t)
                    test[source] = {k: m[k] for k in ("recall", "fpr", "f1", "precision", "pr_auc")}
                    if source in ("deepset", "massive", "massive_dev"):
                        test[source]["by_language"] = {}
                        for l in sorted({lang(rs[i]) for i in idx}):
                            li = [i for i in idx if lang(rs[i]) == l]
                            ml = metrics([rs[i]["y"] for i in li], ps[li], t)
                            test[source]["by_language"][l] = {"n": len(li), "recall": ml["recall"], "fpr": ml["fpr"]}
                    if source in ("deepset", "hackaprompt", "jailbreakllms", "tensor_trust"):
                        qi = [i for i in idx if rs[i]["y"] == 1 and is_question(rs[i]["text"])]
                        ni = [i for i in idx if rs[i]["y"] == 1 and not is_question(rs[i]["text"])]
                        test[source]["attack_recall_question"] = round(float(np.mean(ps[qi] >= t)), 4) if qi else None
                        test[source]["attack_recall_non_question"] = round(float(np.mean(ps[ni] >= t)), 4) if ni else None
            entry["strategies"][strategy] = {"threshold": t, "dev_strata": dev_scores, "dev_worst_stratum": worst(dev_scores),
                                             "test": test}
        best = max(entry["strategies"], key=lambda s: (entry["strategies"][s]["dev_worst_stratum"], s == "T1"))
        entry["dev_selected_strategy"] = best
        report["models"][name] = entry
        print(name, {s: (v["threshold"], v["dev_worst_stratum"], v["test"]["deepset"]["recall"], v["test"]["deepset"]["fpr"],
                         v["test"]["massive_dev"]["fpr"]) for s, v in entry["strategies"].items()}, "->", best, flush=True)
        del model
    # Config-level summary per strategy (mean over seeds).
    summary = {}
    for config in ("H1", "A4"):
        for strategy in ("T1", "T2", "T3", "T4"):
            runs = [report["models"][f"{config}_s{s}"]["strategies"][strategy] for s in (42, 43, 44)]
            pick = lambda f: [f(r) for r in runs if f(r) is not None]
            fields = {"dev_worst_stratum": lambda r: r["dev_worst_stratum"],
                      "deepset_recall": lambda r: r["test"]["deepset"]["recall"], "deepset_fpr": lambda r: r["test"]["deepset"]["fpr"],
                      "deepset_de_recall": lambda r: r["test"]["deepset"]["by_language"].get("de*", {}).get("recall"),
                      "deepset_question_recall": lambda r: r["test"]["deepset"]["attack_recall_question"],
                      "gandalf_recall": lambda r: r["test"]["gandalf"]["recall"],
                      "massive_dev_fpr": lambda r: r["test"]["massive_dev"]["fpr"],
                      "tensor_trust_recall": lambda r: r["test"]["tensor_trust"]["recall"],
                      "jailbreakllms_recall": lambda r: r["test"]["jailbreakllms"]["recall"],
                      "jailbreakllms_fpr": lambda r: r["test"]["jailbreakllms"]["fpr"],
                      "hackaprompt_recall": lambda r: r["test"]["hackaprompt"]["recall"],
                      "security_docs_fpr": lambda r: r["test"]["security_docs"]["fpr"]}
            summary[f"{config}_{strategy}"] = {k: {"mean": round(statistics.mean(v), 4), "sd": round(statistics.stdev(v), 4)}
                                                for k, f in fields.items() for v in [pick(f)] if len(v) > 1}
    report["summary_mean_over_seeds"] = summary
    report["dev_selected_strategy_by_model"] = {k: v["dev_selected_strategy"] for k, v in report["models"].items()}
    (ROOT / "reports/v5_iter4_threshold_ablation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
