"""V5 iteration 2, phase 2: DEV-only selection -> lock -> one-shot TEST1/TEST2/OOD benign.

Rules are the ones in reports/benign_repair_plan.md (copied into the lock file).
E6 (M0) is scored with its frozen checkpoint and threshold as the reference.
A second run refuses to rescore TEST.

Run: .venv/bin/python scripts/finalize_v5_benign_repair.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v5_benign_repair_set import LENGTH_BUCKETS  # noqa: E402
from diagnose_benign_root_cause import guess_language  # noqa: E402
from evaluate_v5_evidence import fit_temperature, metrics, scale  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from train_v5_ablation import BACKBONE, MIXED, TOKENIZER, load_split, predict, tokenize  # noqa: E402
from train_v5_benign_repair import CHECKPOINTS, CONTEXT, NEW_SOURCES, OUT, PREDICTIONS, RUNS, load_rows  # noqa: E402
from trustlaya.utils import device  # noqa: E402
from trustlaya.v5_model import V5Classifier  # noqa: E402

LOCK = ROOT / "reports/v5_benign_repair_lock.json"
MASTER = ROOT / "reports/v5_benign_repair_master.json"
E6 = "E6_mixed_balanced_focal_510"
E6_CKPT = ROOT / f"models/v5-ablation/{E6}/model.safetensors"
E6_SELECTION = json.loads((ROOT / f"reports/experiments/{E6}/selection.json").read_text())
E6_THRESHOLD, E6_TEMPERATURE = E6_SELECTION["threshold"], E6_SELECTION["temperature"]
E6_TEST = {"tensor_trust_recall": 0.991, "jailbreakllms_recall": 0.779, "jailbreakllms_fpr": 0.115, "deepset_fpr": 0.707}
V2_TEST = {"tensor_trust_recall": 0.443, "jailbreakllms_recall": 0.901, "jailbreakllms_fpr": 0.875, "deepset_fpr": 0.364}
BENIGN_SOURCES_DEV = ("jailbreakllms", "security_docs", "arxiv_abstracts", "dolly", "oasst2", "aya")
PLAN_RULES = (ROOT / "reports/benign_repair_plan.md").read_text()


def language(r):
    if r.get("language") in ("en", "de", "tr"):
        return r["language"]
    return guess_language(r["text"]) + "*" if r["source"] == "deepset" else "unknown"


def bucket(n):
    return next(name for name, low, high in LENGTH_BUCKETS if low <= n <= high)


def groups_accuracy(rows, probs, threshold):
    accs = defaultdict(list)
    for r, p in zip(rows, probs):
        accs[(r["source"], r["y"])].append((p >= threshold) == bool(r["y"]))
    return float(np.mean([np.mean(v) for v in accs.values()]))


def report(rows, probs, threshold, temperature):
    probs = np.asarray(probs)
    y = [r["y"] for r in rows]
    out = {"pooled": metrics(y, probs, threshold),
           "pooled_calibrated": {k: metrics(y, scale(probs, temperature))[k] for k in ("ece", "brier")},
           "per_source": {}, "language_x_length": {}}
    for source in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == source]
        out["per_source"][source] = metrics([y[i] for i in idx], probs[idx], threshold)
        cells = defaultdict(list)
        for i in idx:
            cells[(language(rows[i]), bucket(len(rows[i]["tokens"])))].append(i)
            cells[(language(rows[i]), "all")].append(i)
        out["language_x_length"][source] = {
            f"{lang}/{b}": {**{k: m[k] for k in ("n", "fpr", "recall", "f1")},
                            "mean_score": round(float(probs[ids].mean()), 4)}
            for (lang, b), ids in sorted(cells.items()) for m in [metrics([y[i] for i in ids], probs[ids], threshold)]}
    return out


def load_model(path):
    model = V5Classifier(BACKBONE, pretrained=False)
    model.load(path)
    return model.to(device())


def main():
    if MASTER.exists() and json.loads(MASTER.read_text()).get("test_evaluated"):
        sys.exit("TEST was already evaluated once; refusing to rescore.")
    rows = load_rows()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", set(MIXED) | set(NEW_SOURCES)))
    e6_dev, _ = predict(load_model(E6_CKPT), dev_rows, CONTEXT, tokenizer, device())
    e6_rep = report(dev_rows, e6_dev, E6_THRESHOLD, E6_TEMPERATURE)
    e6_recall = {s: e6_rep["per_source"][s]["recall"] for s in ("tensor_trust", "jailbreakllms")}

    choices = {"M0_E6_reference": {"threshold": E6_THRESHOLD, "temperature": E6_TEMPERATURE, "frozen_from": f"reports/experiments/{E6}/selection.json",
                                   "dev": e6_rep, "checkpoint_sha256": sha256(E6_CKPT)}}
    grid = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]
    for run_id in RUNS:
        ckpt_file = OUT / run_id / "checkpoint.sha256"
        if not ckpt_file.exists():
            choices[run_id] = {"status": "not_trained"}
            continue
        probs = np.load(PREDICTIONS / f"{run_id}_DEV.npy")
        scores = {t: groups_accuracy(dev_rows, probs, t) for t in grid}
        threshold = max(grid, key=lambda t: (scores[t], -abs(t - 0.5)))
        temperature = fit_temperature([r["y"] for r in dev_rows], probs)
        rep = report(dev_rows, probs, threshold, temperature)
        recall = {s: rep["per_source"][s]["recall"] for s in ("tensor_trust", "jailbreakllms")}
        benign_fpr = {s: rep["per_source"][s]["fpr"] for s in BENIGN_SOURCES_DEV}
        choices[run_id] = {
            "threshold": threshold, "temperature": round(temperature, 4), "dev_macro_group_accuracy": round(scores[threshold], 4),
            "dev_attack_recall": recall, "dev_benign_fpr": benign_fpr,
            "dev_mean_benign_fpr": round(float(np.mean(list(benign_fpr.values()))), 4),
            "dev_mean_attack_recall": round(float(np.mean(list(recall.values()))), 4),
            "dev_pr_auc": rep["pooled"]["pr_auc"], "dev_ece_calibrated": rep["pooled_calibrated"]["ece"],
            "REGRESSION": any(recall[s] < e6_recall[s] - 0.05 for s in recall),
            "checkpoint_sha256": ckpt_file.read_text().split()[0], "dev": rep}
        (OUT / run_id / "selection.json").write_text(json.dumps({k: v for k, v in choices[run_id].items() if k != "dev"}, indent=2) + "\n")
        (OUT / run_id / "source_metrics.json").write_text(json.dumps(
            {**json.loads((OUT / run_id / "source_metrics.json").read_text()), "DEV_selected_threshold": rep}, indent=2) + "\n")
    eligible = {k: v for k, v in choices.items() if "dev_mean_benign_fpr" in v and not v["REGRESSION"]}
    candidate = min(eligible, key=lambda k: (eligible[k]["dev_mean_benign_fpr"], -eligible[k]["dev_mean_attack_recall"],
                                             -(eligible[k]["dev_pr_auc"] or 0), eligible[k]["dev_ece_calibrated"])) if eligible else None
    lock = {"frozen_at": datetime.now(timezone.utc).isoformat(), "plan_sha256": sha256(ROOT / "reports/benign_repair_plan.md"),
            "e6_dev_recall_reference": e6_recall, "dev_selected_candidate": candidate,
            "selections": {k: {kk: vv for kk, vv in v.items() if kk != "dev"} for k, v in choices.items()}}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"selection frozen: candidate {candidate}", flush=True)

    # One-shot TEST for every run and the E6 reference.
    test1 = tokenize(tokenizer, load_split(rows, "TEST", set(MIXED) | set(NEW_SOURCES)))
    test2 = tokenize(tokenizer, load_split(rows, "OOD_TEST", {"deepset", "gandalf"}))
    ood_benign = tokenize(tokenizer, load_split(rows, "OOD_BENIGN_TEST", {"massive"}))
    results = {}
    runs = [("M0_E6_reference", E6_CKPT)] + [(k, CHECKPOINTS / k / "model.safetensors") for k in RUNS if "threshold" in choices[k]]
    for run_id, path in runs:
        if sha256(path) != choices[run_id]["checkpoint_sha256"]:
            sys.exit(f"{run_id}: checkpoint changed after selection")
        model = load_model(path)
        threshold = choices[run_id]["threshold"]
        temperature = choices[run_id].get("temperature", 1.0)
        results[run_id] = {name: report(subset, predict(model, subset, CONTEXT, tokenizer, device())[0], threshold, temperature)
                           for name, subset in (("TEST1", test1), ("TEST2", test2), ("TEST2_OOD_BENIGN", ood_benign))}
        if run_id != "M0_E6_reference":
            (OUT / run_id / "metrics_test.json").write_text(json.dumps(results[run_id], indent=2) + "\n")
        print(f"{run_id}: TEST scored", flush=True)
        del model

    def key_metrics(res):
        t1, t2, ob = res["TEST1"]["per_source"], res["TEST2"]["per_source"], res["TEST2_OOD_BENIGN"]
        lang = {k.split("/")[0]: v["fpr"] for k, v in ob["language_x_length"]["massive"].items() if k.endswith("/all")}
        return {"tensor_trust_recall": t1["tensor_trust"]["recall"], "jailbreakllms_recall": t1["jailbreakllms"]["recall"],
                "jailbreakllms_fpr": t1["jailbreakllms"]["fpr"], "security_docs_fpr": t1["security_docs"]["fpr"],
                "arxiv_fpr": t1["arxiv_abstracts"]["fpr"], "deepset_fpr": t2["deepset"]["fpr"],
                "deepset_recall": t2["deepset"]["recall"], "gandalf_recall": t2["gandalf"]["recall"],
                "massive_fpr": ob["per_source"]["massive"]["fpr"], "massive_fpr_by_language": lang,
                "new_source_fpr": {s: t1[s]["fpr"] for s in NEW_SOURCES if s in t1},
                "test1_ece_calibrated": res["TEST1"]["pooled_calibrated"]["ece"], "test1_ece_raw": res["TEST1"]["pooled"]["ece"]}

    summary = {k: key_metrics(v) for k, v in results.items()}
    e6 = summary["M0_E6_reference"]
    status, flags = "NO_GO", {}
    if candidate:
        c = summary[candidate]
        A = c["deepset_fpr"] <= 0.40 and c["deepset_fpr"] <= e6["deepset_fpr"] - 0.25
        B = (c["massive_fpr"] <= 0.15 and all(v <= 0.25 for v in c["massive_fpr_by_language"].values())
             and c["massive_fpr"] < e6["massive_fpr"])
        C = c["tensor_trust_recall"] >= 0.95
        D = c["jailbreakllms_recall"] >= 0.70
        E = c["security_docs_fpr"] <= 0.10 and c["arxiv_fpr"] <= 0.10
        F = c["test1_ece_calibrated"] <= 0.05
        flags = {"A_deepset": A, "B_massive": B, "C_tensor_trust": C, "D_jailbreak": D, "E_docs_arxiv": E,
                 "F_calibration": F, "G_not_deepset_only": B,
                 "GENERALIZATION_FAILURE": c["deepset_fpr"] < e6["deepset_fpr"] and c["massive_fpr"] > e6["massive_fpr"]}
        deepset_gain = e6["deepset_fpr"] - c["deepset_fpr"]
        massive_gain = e6["massive_fpr"] - c["massive_fpr"]
        if not (C and D):
            status = "REGRESSION"
        elif A and B and E and F:
            status = "READY_FOR_NEW_EXTERNAL_VALIDATION"
        elif B and E and F and deepset_gain >= 0.15:
            status = "GENERALIZATION_IMPROVED"
        elif deepset_gain >= 0.10 or massive_gain >= 0.10:
            status = "PARTIAL"
    master = {"test_evaluated": True, "evaluated_at": datetime.now(timezone.utc).isoformat(), "lock_sha256": sha256(LOCK),
              "dev_selected_candidate": candidate, "V5_STATUS": status, "status_flags": flags,
              "v2_default": True, "firewall_model": "V2", "reference_E6_test_v1": E6_TEST, "reference_V2_test_v1": V2_TEST,
              "summary": summary, "selections": lock["selections"],
              "training": {k: json.loads((OUT / k / "training_log.json").read_text())["wall_clock_training_s"]
                           for k in RUNS if (OUT / k / "training_log.json").exists()},
              "details": results}
    MASTER.write_text(json.dumps(master, indent=2) + "\n")
    print(f"V5_STATUS = {status} (candidate {candidate})")


if __name__ == "__main__":
    main()
