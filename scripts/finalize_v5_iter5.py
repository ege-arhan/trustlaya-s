"""V5 iteration 5 finalization (plan: reports/iter5_plan.md).

  --phase select : DEV-only thresholds for S1-S5 (seed 42); writes the two DEV-best run names to
                   reports/v5_iter5_dev_selection.json (they then get seeds 43/44).
  --phase test   : locks thresholds/temperatures/checkpoints for every run (S0 = iteration-3 H1
                   seeds, S1-S5 seed 42, the two candidates at seeds 43/44), then scores TEST1,
                   TEST2 (deepset, Gandalf), MASSIVE test and MASSIVE dev once.

Run: .venv/bin/python scripts/finalize_v5_iter5.py --phase select|test
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v5_evidence import fit_temperature, metrics  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from finalize_v5_benign_repair import report  # noqa: E402
from finalize_v5_iter3 import balanced_threshold  # noqa: E402
from threshold_ablation_v5_iter4 import is_question, lang  # noqa: E402
from train_v5_ablation import BACKBONE, MIXED, TOKENIZER, load_split, predict, tokenize  # noqa: E402
from train_v5_benign_repair import CONTEXT, NEW_SOURCES  # noqa: E402
from train_v5_iter3 import DEV_SOURCES  # noqa: E402
from train_v5_iter5 import RUNS, rows_iter5  # noqa: E402
from trustlaya.utils import device  # noqa: E402
from trustlaya.v5_model import V5Classifier  # noqa: E402

SELECTION = ROOT / "reports/v5_iter5_dev_selection.json"
LOCK = ROOT / "reports/v5_iter5_lock.json"
MASTER = ROOT / "reports/v5_iter5_master.json"
S0 = {f"S0_s{s}": f"models/v5-iter3/H1_grouped_s{s}" for s in (42, 43, 44)}


def load(folder):
    model = V5Classifier(BACKBONE, pretrained=False)
    model.load(ROOT / folder / "model.safetensors")
    return model.to(device())


def dev_choice(model, dev_rows, tokenizer):
    probs, _ = predict(model, dev_rows, CONTEXT, tokenizer, device())
    t = balanced_threshold(dev_rows, probs)
    y = [r["y"] for r in dev_rows]
    groups = {}
    for r, p in zip(dev_rows, probs):
        groups.setdefault((r["source"], r["y"]), []).append((p >= t) == bool(r["y"]))
    attack = np.mean([np.mean(v) for g, v in groups.items() if g[1]])
    benign = np.mean([np.mean(v) for g, v in groups.items() if not g[1]])
    return {"threshold": t, "temperature": round(fit_temperature(y, probs), 4),
            "dev_class_balanced_accuracy": round(float((attack + benign) / 2), 4),
            "dev_pr_auc": metrics(y, probs, t)["pr_auc"]}


def select():
    rows = rows_iter5()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", DEV_SOURCES))
    out = {}
    for run_id in RUNS:
        folder = f"models/v5-iter5/{run_id}_s42"
        out[run_id] = {**dev_choice(load(folder), dev_rows, tokenizer), "checkpoint_sha256": sha256(ROOT / folder / "model.safetensors")}
        print(run_id, out[run_id], flush=True)
    top = sorted(out, key=lambda k: (out[k]["dev_class_balanced_accuracy"], out[k]["dev_pr_auc"] or 0), reverse=True)[:2]
    SELECTION.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(), "rule": "DEV class-balanced accuracy at T1, tie DEV PR-AUC",
                                     "runs_seed42": out, "top2_for_seeds_43_44": top}, indent=2) + "\n")
    print("top2", top)


def summary(res, test1, probs1, t):
    t1, t2 = res["TEST1"]["per_source"], res["TEST2"]["per_source"]
    ds = [(r, p) for r, p in zip(res["_rows_TEST2"], res["_probs_TEST2"]) if r["source"] == "deepset"]
    def bucket(pred):
        idx = [(r, p) for r, p in ds if pred(r)]
        if not idx:
            return None
        m = metrics([r["y"] for r, _ in idx], [p for _, p in idx], t)
        return {k: m[k] for k in ("n", "positive", "recall", "precision", "fpr", "pr_auc")}
    return {"tensor_trust_recall": t1["tensor_trust"]["recall"], "jailbreakllms_recall": t1["jailbreakllms"]["recall"],
            "jailbreakllms_fpr": t1["jailbreakllms"]["fpr"], "hackaprompt_recall": t1["hackaprompt"]["recall"],
            "security_docs_fpr": t1["security_docs"]["fpr"], "arxiv_fpr": t1["arxiv_abstracts"]["fpr"],
            "new_benign_fpr": {s: t1[s]["fpr"] for s in NEW_SOURCES},
            "deepset_recall": t2["deepset"]["recall"], "deepset_fpr": t2["deepset"]["fpr"], "deepset_pr_auc": t2["deepset"]["pr_auc"],
            "gandalf_recall": t2["gandalf"]["recall"],
            "massive_test_fpr": res["MASSIVE_TEST"]["per_source"]["massive"]["fpr"],
            "massive_dev_fpr": res["MASSIVE_DEV"]["per_source"]["massive_dev"]["fpr"],
            "massive_dev_fpr_by_language": {k.split("/")[0]: v["fpr"] for k, v in res["MASSIVE_DEV"]["language_x_length"]["massive_dev"].items() if k.endswith("/all")},
            "test1_ece_calibrated": res["TEST1"]["pooled_calibrated"]["ece"], "test1_ece_raw": res["TEST1"]["pooled"]["ece"],
            "deepset_buckets": {"german*": bucket(lambda r: lang(r) == "de*"), "english*": bucket(lambda r: lang(r) == "en*"),
                                "question": bucket(lambda r: is_question(r["text"])), "non_question": bucket(lambda r: not is_question(r["text"])),
                                "short_<=94": bucket(lambda r: len(r["tokens"]) <= 94), "long_>94": bucket(lambda r: len(r["tokens"]) > 94)}}


def aggregate(items):
    out = {}
    for key, value in items[0].items():
        if isinstance(value, (int, float)) and value is not None:
            vals = [i[key] for i in items if i[key] is not None]
            out[key] = {"mean": round(statistics.mean(vals), 4), "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
                        "min": round(min(vals), 4), "max": round(max(vals), 4)}
    return out


def test():
    if MASTER.exists() and json.loads(MASTER.read_text()).get("test_evaluated"):
        sys.exit("TEST already evaluated once; refusing to rescore.")
    selection = json.loads(SELECTION.read_text())
    runs = dict(S0)
    runs.update({f"{k}_s42": f"models/v5-iter5/{k}_s42" for k in RUNS})
    for k in selection["top2_for_seeds_43_44"]:
        runs.update({f"{k}_s{s}": f"models/v5-iter5/{k}_s{s}" for s in (43, 44)})
    rows = rows_iter5()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", DEV_SOURCES))
    sets = {"TEST1": load_split(rows, "TEST", set(MIXED) | set(NEW_SOURCES) | {"hackaprompt"}),
            "TEST2": load_split(rows, "OOD_TEST", {"deepset", "gandalf"}),
            "MASSIVE_TEST": load_split(rows, "OOD_BENIGN_TEST", {"massive"}),
            "MASSIVE_DEV": load_split(rows, "OOD_BENIGN_FRESH", {"massive_dev"})}
    assert not any(r.get("synthetic") for s in sets.values() for r in s), "synthetic rows must never be scored as TEST"
    for subset in sets.values():
        tokenize(tokenizer, subset)
    models, choices = {}, {}
    for name, folder in runs.items():
        models[name] = load(folder)
        choices[name] = {**dev_choice(models[name], dev_rows, tokenizer), "checkpoint_sha256": sha256(ROOT / folder / "model.safetensors")}
    LOCK.write_text(json.dumps({"frozen_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha256(ROOT / "reports/iter5_plan.md"),
                                "runs": choices}, indent=2) + "\n")
    print("locked", {k: v["threshold"] for k, v in choices.items()}, flush=True)
    summaries = {}
    for name, model in models.items():
        c = choices[name]
        probs = {k: predict(model, s, CONTEXT, tokenizer, device())[0] for k, s in sets.items()}
        res = {k: report(sets[k], probs[k], c["threshold"], c["temperature"]) for k in sets}
        res["_rows_TEST2"], res["_probs_TEST2"] = sets["TEST2"], probs["TEST2"]
        summaries[name] = summary(res, sets["TEST1"], probs["TEST1"], c["threshold"])
        print(name, "scored", flush=True)
    configs = {"S0": [f"S0_s{s}" for s in (42, 43, 44)]}
    for k in selection["top2_for_seeds_43_44"]:
        configs[k] = [f"{k}_s{s}" for s in (42, 43, 44)]
    agg = {c: aggregate([summaries[n] for n in names]) for c, names in configs.items()}
    cand = max(selection["top2_for_seeds_43_44"], key=lambda k: selection["runs_seed42"][k]["dev_class_balanced_accuracy"])
    c, s0 = agg[cand], agg["S0"]
    m = lambda d, k: d[k]["mean"]
    rules = {"A_deepset_recall_ge_0.40": m(c, "deepset_recall") >= 0.40, "B_deepset_fpr_le_0.15": m(c, "deepset_fpr") <= 0.15,
             "C_massive_not_worse": m(c, "massive_dev_fpr") <= m(s0, "massive_dev_fpr") + 0.03 and m(c, "massive_test_fpr") <= m(s0, "massive_test_fpr") + 0.03,
             "D_tensor_trust_ge_0.95": m(c, "tensor_trust_recall") >= 0.95, "E_jailbreak_ge_0.75": m(c, "jailbreakllms_recall") >= 0.75,
             "F_independent_multilingual_attack_source": "not assessable (no real source exists)",
             "G_seed_sd_deepset_recall_le_0.05": (c["deepset_recall"]["sd"] or 0) <= 0.05}
    status = "PARTIAL" if all(v is True for k, v in rules.items() if not k.startswith("F")) else "NO_GO"
    MASTER.write_text(json.dumps({"test_evaluated": True, "evaluated_at": datetime.now(timezone.utc).isoformat(),
                                  "lock_sha256": sha256(LOCK), "dev_selected_candidate": cand, "V5_STATUS": status,
                                  "status_note": "PARTIAL here means a benchmark-specific effect; GENERALIZATION_IMPROVED is unreachable without an independent multilingual attack source",
                                  "rules": rules, "config_means": agg, "per_run": summaries, "v2_default": True, "firewall_model": "V2"},
                                 indent=2) + "\n")
    print("V5_STATUS", status, "candidate", cand)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "test"), required=True)
    select() if parser.parse_args().phase == "select" else test()
