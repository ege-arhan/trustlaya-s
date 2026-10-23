"""V5 iteration 3: DEV-only thresholds -> lock -> one-shot TEST (plan: reports/iter3_plan.md).

Class-balanced DEV threshold per run. The A4 control seeds (variance study) are
re-thresholded with the same class-balanced rule on the iteration-3 DEV, so the
comparison uses one rule. Verdict = mean over the three H1_grouped seeds.

Run: .venv/bin/python scripts/finalize_v5_iter3.py
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v5_evidence import fit_temperature, metrics, scale  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from finalize_v5_benign_repair import report  # noqa: E402
from train_v5_ablation import BACKBONE, MIXED, TOKENIZER, load_split, predict, tokenize  # noqa: E402
from train_v5_benign_repair import CONTEXT, NEW_SOURCES  # noqa: E402
from train_v5_iter3 import DEV_SOURCES, RUNS, rows_iter3  # noqa: E402
from trustlaya.utils import device  # noqa: E402
from trustlaya.v5_model import V5Classifier  # noqa: E402

LOCK = ROOT / "reports/v5_iter3_lock.json"
MASTER = ROOT / "reports/v5_iter3_master.json"
MODELS = {**{k: f"models/v5-iter3/{k}" for k in RUNS},
          "A4control_s42": "models/v5-benign-repair/A4_multilingual_4x",
          "A4control_s43": "models/v5-variance/A4data_grouped_s43",
          "A4control_s44": "models/v5-variance/A4data_grouped_s44"}
GRID = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]


def balanced_threshold(rows, probs):
    groups = defaultdict(list)
    for r, p in zip(rows, probs):
        groups[(r["source"], r["y"])].append(p)
    def score(t):
        acc = {g: np.mean([(p >= t) == bool(g[1]) for p in v]) for g, v in groups.items()}
        return (np.mean([v for g, v in acc.items() if g[1] == 1]) + np.mean([v for g, v in acc.items() if g[1] == 0])) / 2
    return max(GRID, key=lambda t: (score(t), -abs(t - 0.5)))


def summary(res, test1, probs_t1, threshold):
    t1, t2 = res["TEST1"]["per_source"], res["TEST2"]["per_source"]
    hp = [(r, p) for r, p in zip(test1, probs_t1) if r["source"] == "hackaprompt"]
    pwn = [p for r, p in hp if re.search(r"pwn", r["text"], re.I)]
    nopwn = [p for r, p in hp if not re.search(r"pwn", r["text"], re.I)]
    fresh = res["FRESH_MASSIVE_DEV"]["language_x_length"]["massive_dev"]
    massive = res["MASSIVE_TEST"]["language_x_length"]["massive"]
    return {
        "tensor_trust_recall": t1["tensor_trust"]["recall"], "jailbreakllms_recall": t1["jailbreakllms"]["recall"],
        "jailbreakllms_fpr": t1["jailbreakllms"]["fpr"], "hackaprompt_recall": t1["hackaprompt"]["recall"],
        "hackaprompt_recall_with_pwn": round(float(np.mean([p >= threshold for p in pwn])), 4),
        "hackaprompt_recall_without_pwn": round(float(np.mean([p >= threshold for p in nopwn])), 4),
        "security_docs_fpr": t1["security_docs"]["fpr"], "arxiv_fpr": t1["arxiv_abstracts"]["fpr"],
        "new_benign_fpr": {s: t1[s]["fpr"] for s in NEW_SOURCES},
        "deepset_recall": t2["deepset"]["recall"], "deepset_fpr": t2["deepset"]["fpr"],
        "deepset_pr_auc": t2["deepset"]["pr_auc"], "gandalf_recall": t2["gandalf"]["recall"],
        "massive_test_fpr": res["MASSIVE_TEST"]["per_source"]["massive"]["fpr"],
        "massive_test_fpr_by_language": {k.split("/")[0]: v["fpr"] for k, v in massive.items() if k.endswith("/all")},
        "fresh_massive_dev_fpr": res["FRESH_MASSIVE_DEV"]["per_source"]["massive_dev"]["fpr"],
        "fresh_massive_dev_fpr_by_language": {k.split("/")[0]: v["fpr"] for k, v in fresh.items() if k.endswith("/all")},
        "test1_ece_raw": res["TEST1"]["pooled"]["ece"], "test1_ece_calibrated": res["TEST1"]["pooled_calibrated"]["ece"]}


def aggregate(items):
    out = {}
    for key, value in items[0].items():
        if isinstance(value, dict):
            out[key] = {k: round(statistics.mean(i[key][k] for i in items), 4) for k in value}
        elif value is not None:
            vals = [i[key] for i in items if i[key] is not None]
            out[key] = {"mean": round(statistics.mean(vals), 4),
                        "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None}
    return out


def main():
    if MASTER.exists() and json.loads(MASTER.read_text()).get("test_evaluated"):
        sys.exit("TEST already evaluated once; refusing to rescore.")
    rows = rows_iter3()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", DEV_SOURCES))
    sets = {"TEST1": load_split(rows, "TEST", set(MIXED) | set(NEW_SOURCES) | {"hackaprompt"}),
            "TEST2": load_split(rows, "OOD_TEST", {"deepset", "gandalf"}),
            "MASSIVE_TEST": load_split(rows, "OOD_BENIGN_TEST", {"massive"}),
            "FRESH_MASSIVE_DEV": load_split(rows, "OOD_BENIGN_FRESH", {"massive_dev"})}
    for subset in sets.values():
        tokenize(tokenizer, subset)
    dev = device()
    models, choices = {}, {}
    for run_id, folder in MODELS.items():
        path = ROOT / folder / "model.safetensors"
        if not path.exists():
            choices[run_id] = {"status": "missing"}
            continue
        model = V5Classifier(BACKBONE, pretrained=False)
        model.load(path)
        models[run_id] = model.to(dev)
        probs, _ = predict(models[run_id], dev_rows, CONTEXT, tokenizer, dev)
        threshold = balanced_threshold(dev_rows, probs)
        temperature = fit_temperature([r["y"] for r in dev_rows], probs)
        dev_rep = report(dev_rows, probs, threshold, temperature)
        choices[run_id] = {"threshold": threshold, "temperature": round(temperature, 4),
                           "checkpoint_sha256": sha256(path), "dev_per_source": {
                               s: {k: m[k] for k in ("recall", "fpr", "f1", "pr_auc")} for s, m in dev_rep["per_source"].items()}}
    lock = {"frozen_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha256(ROOT / "reports/iter3_plan.md"),
            "threshold_rule": "class-balanced DEV accuracy", "runs": choices}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print("thresholds frozen", {k: v.get("threshold") for k, v in choices.items()}, flush=True)

    results, summaries = {}, {}
    for run_id, model in models.items():
        c = choices[run_id]
        probs = {name: predict(model, subset, CONTEXT, tokenizer, dev)[0] for name, subset in sets.items()}
        results[run_id] = {name: report(sets[name], probs[name], c["threshold"], c["temperature"]) for name in sets}
        summaries[run_id] = summary(results[run_id], sets["TEST1"], probs["TEST1"], c["threshold"])
        print(f"{run_id}: TEST scored", flush=True)
    h1 = aggregate([summaries[k] for k in ("H1_grouped_s42", "H1_grouped_s43", "H1_grouped_s44") if k in summaries])
    control = aggregate([summaries[k] for k in ("A4control_s42", "A4control_s43", "A4control_s44") if k in summaries])
    m = lambda key: h1[key]["mean"]
    fresh_lang = h1["fresh_massive_dev_fpr_by_language"]
    rules = {
        "deepset": m("deepset_recall") >= 0.60 and m("deepset_fpr") <= 0.30,
        "fresh_massive": m("fresh_massive_dev_fpr") <= 0.20 and all(v <= 0.30 for v in fresh_lang.values()),
        "tensor_trust": m("tensor_trust_recall") >= 0.95, "jailbreakllms": m("jailbreakllms_recall") >= 0.70,
        "hackaprompt": m("hackaprompt_recall") >= 0.90 and m("hackaprompt_recall_without_pwn") >= 0.80,
        "docs_arxiv": m("security_docs_fpr") <= 0.10 and m("arxiv_fpr") <= 0.10,
        "calibration": m("test1_ece_calibrated") <= 0.05,
    }
    attack_rules = ("tensor_trust", "jailbreakllms", "hackaprompt")
    benign_fails = [k for k in ("fresh_massive", "docs_arxiv") if not rules[k]] + (
        ["deepset_fpr"] if m("deepset_fpr") > 0.30 else [])
    if m("tensor_trust_recall") < 0.95 or m("jailbreakllms_recall") < 0.70 or m("deepset_recall") < 0.50:
        status = "REGRESSION"
    elif all(rules.values()):
        status = "READY_FOR_NEW_EXTERNAL_VALIDATION"
    elif all(rules[k] for k in attack_rules) and m("deepset_recall") >= 0.60 and rules["calibration"] and len(benign_fails) <= 1:
        status = "GENERALIZATION_IMPROVED"
    elif (m("deepset_recall") - control["deepset_recall"]["mean"] >= 0.10
          or control["fresh_massive_dev_fpr"]["mean"] - m("fresh_massive_dev_fpr") >= 0.10):
        status = "PARTIAL"
    else:
        status = "NO_GO"
    master = {"test_evaluated": True, "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "lock_sha256": sha256(LOCK), "V5_STATUS": status, "rules": rules,
              "v2_default": True, "firewall_model": "V2",
              "H1_grouped_mean": h1, "A4_control_mean": control, "per_run": summaries,
              "note": "deepset, Gandalf and MASSIVE test were scored in earlier iterations (reused); "
                      "MASSIVE dev is the only fresh OOD benign test.",
              "details": results}
    MASTER.write_text(json.dumps(master, indent=2) + "\n")
    print(f"V5_STATUS = {status}")


if __name__ == "__main__":
    main()
