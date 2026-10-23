"""V5 ablation phase 2: freeze DEV-only choices, then evaluate TEST1/TEST2 exactly once.

1. For every trained experiment, on DEV only: threshold (rule in
   train_v5_ablation.SELECTION_RULES), temperature, DEV metrics. The DEV-selected
   candidate and the status rules below are written to
   reports/v5_selection_lock.json with checksums *before* any TEST row is scored.
2. TEST1 (pool TEST: Tensor Trust, JailbreakLLMs, security docs, arXiv) and
   TEST2/OOD (deepset, Gandalf; JailbreakBench as a task-mismatch side report)
   are scored once for all experiments. A second run refuses to rescore.

Run: .venv/bin/python scripts/finalize_v5_ablation.py
"""

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v5_evidence import fit_temperature, metrics, scale  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from train_v5_ablation import (BACKBONE, CHECKPOINTS, EXPERIMENTS, EXPERIMENTS_DIR, MIXED,  # noqa: E402
                               PREDICTIONS, SELECTION_RULES, TOKENIZER, load_split, predict, tokenize)
from trustlaya.utils import device, normalize  # noqa: E402
from trustlaya.v5_model import V5Classifier  # noqa: E402

LOCK = ROOT / "reports/v5_selection_lock.json"
MASTER = ROOT / "reports/v5_ablation_master.json"
GROUPS = [("tensor_trust", 1), ("jailbreakllms", 1), ("jailbreakllms", 0), ("security_docs", 0), ("arxiv_abstracts", 0)]
BUCKETS = (("<=94", 0, 94), ("95-256", 95, 256), ("257-510", 257, 510), (">510", 511, 10 ** 9))
# Written before TEST is scored; copied into the lock file.
STATUS_RULES = {
    "generalization_failure": "at the frozen threshold, on TEST1+TEST2 attack sources (tensor_trust, jailbreakllms, "
                              "deepset, gandalf): min recall < 0.5 while max recall >= 0.8; or any benign source "
                              "(jailbreakllms, security_docs, arxiv_abstracts, deepset) FPR > 0.20",
    "beats_v2": "candidate TEST1 JailbreakLLMs F1 and deepset F1 both above frozen V2, and Tensor Trust and Gandalf recall "
                "both above frozen V2, with every benign-source FPR below V2's",
    "NO_GO": "candidate does not beat V2 (rule beats_v2)",
    "EXPERIMENTAL": "beats V2 but generalization_failure",
    "PROMISING": "beats V2, no generalization_failure, but max benign FPR > 0.05 or calibrated TEST ECE > 0.10",
    "READY_FOR_EXTERNAL_VALIDATION": "beats V2, no generalization_failure, max benign FPR <= 0.05, calibrated TEST ECE <= 0.10",
    "never": "PRODUCTION READY is not a possible outcome of this script",
}


def group_accuracy(rows, probs, threshold):
    accs = []
    for source, label in GROUPS:
        idx = [i for i, r in enumerate(rows) if r["source"] == source and r["y"] == label]
        if idx:
            correct = sum((probs[i] >= threshold) == bool(label) for i in idx)
            accs.append(correct / len(idx))
    return sum(accs) / len(accs)


def breakdown(rows, probs, threshold, temperature):
    probs = np.asarray(probs)
    calibrated = scale(probs, temperature)
    y = [r["y"] for r in rows]
    out = {"pooled": metrics(y, probs, threshold), "pooled_calibrated": metrics(y, calibrated, threshold_on(threshold, temperature)),
           "per_source": {}, "per_length": {}}
    for source in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == source]
        out["per_source"][source] = metrics([y[i] for i in idx], probs[idx], threshold)
        out["per_source"][source]["calibrated_ece"] = metrics([y[i] for i in idx], calibrated[idx])["ece"]
        out["per_source"][source]["calibrated_brier"] = metrics([y[i] for i in idx], calibrated[idx])["brier"]
        buckets = {}
        for name, low, high in BUCKETS:
            sub = [i for i in idx if low <= len(rows[i]["tokens"]) <= high]
            if sub:
                m = metrics([y[i] for i in sub], probs[sub], threshold)
                buckets[name] = {k: m[k] for k in ("n", "positive", "recall", "fpr", "f1")}
        out["per_length"][source] = buckets
    out["pooled"]["accuracy_secondary"] = round((out["pooled"]["tp"] + out["pooled"]["tn"]) / out["pooled"]["n"], 4)
    return out


def threshold_on(threshold, temperature):
    """The frozen raw threshold expressed on the calibrated scale (same decisions)."""
    return float(scale(np.asarray([threshold]), temperature)[0])


def select():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = [json.loads(line) for line in open(ROOT / "data/v5/all_rows.jsonl")]
    dev_rows = tokenize(tokenizer, load_split(rows, "DEV", set(MIXED)))
    choices = {}
    for experiment_id in EXPERIMENTS:
        ckpt_file = EXPERIMENTS_DIR / experiment_id / "checkpoint.sha256"
        if not ckpt_file.exists():
            choices[experiment_id] = {"status": "not_trained"}
            continue
        probs = np.load(PREDICTIONS / f"{experiment_id}_DEV.npy")
        grid = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]
        scores = {t: group_accuracy(dev_rows, probs, t) for t in grid}
        threshold = max(grid, key=lambda t: (scores[t], -abs(t - 0.5)))
        temperature = fit_temperature([r["y"] for r in dev_rows], probs)
        dev = breakdown(dev_rows, probs, threshold, temperature)
        dev_half = breakdown(dev_rows, probs, 0.5, temperature)
        selection = {"threshold": threshold, "dev_macro_group_accuracy": round(scores[threshold], 4),
                     "dev_macro_group_accuracy_at_0.5": round(scores[0.5], 4),
                     "temperature": round(temperature, 4), "temperature_at_search_bound": temperature >= 19.9,
                     "dev_pr_auc": dev["pooled"]["pr_auc"],
                     "checkpoint_sha256": ckpt_file.read_text().split()[0]}
        (EXPERIMENTS_DIR / experiment_id / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        metrics_file = EXPERIMENTS_DIR / experiment_id / "metrics.json"
        stored = json.loads(metrics_file.read_text())
        stored.update(DEV_selected_threshold=dev, DEV_threshold_0_5_breakdown=dev_half)
        metrics_file.write_text(json.dumps(stored, indent=2) + "\n")
        choices[experiment_id] = selection
    trained = {k: v for k, v in choices.items() if "threshold" in v}
    candidate = max(trained, key=lambda k: (trained[k]["dev_macro_group_accuracy"], trained[k]["dev_pr_auc"] or 0))
    lock = {"frozen_at": datetime.now(timezone.utc).isoformat(), "selection_rules": SELECTION_RULES,
            "status_rules": STATUS_RULES, "dev_selected_candidate": candidate, "experiments": choices,
            "selection_files_sha256": {k: sha256(EXPERIMENTS_DIR / k / "selection.json") for k in trained},
            "test_scored_before_lock": False}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    return lock, rows, tokenizer


def test_once(lock, rows, tokenizer):
    if MASTER.exists() and json.loads(MASTER.read_text()).get("test_evaluated"):
        sys.exit("TEST1/TEST2 were already evaluated once; refusing to rescore.")
    test1 = tokenize(tokenizer, load_split(rows, "TEST", set(MIXED)))
    test2 = tokenize(tokenizer, load_split(rows, "OOD_TEST", {"deepset", "gandalf"}))
    jbb = [r for r in rows if r["source"] == "jailbreakbench" and r["status"] == "included"]
    for r in jbb:
        r["tokens"] = tokenizer(normalize(r["text"]), add_special_tokens=False, verbose=False)["input_ids"]
        r["y"] = float(r["label"] == "HARMFUL_REQUEST")
    dev = device()
    results = {}
    for experiment_id, choice in lock["experiments"].items():
        if "threshold" not in choice:
            results[experiment_id] = {"status": "not_trained"}
            continue
        spec = EXPERIMENTS[experiment_id]
        ckpt = CHECKPOINTS / experiment_id / "model.safetensors"
        if sha256(ckpt) != choice["checkpoint_sha256"]:
            sys.exit(f"{experiment_id}: checkpoint changed after selection")
        model = V5Classifier(BACKBONE, pretrained=False)
        model.load(ckpt)
        model.to(dev)
        out = {}
        for name, subset in (("TEST1", test1), ("TEST2", test2), ("JBB_task_mismatch", jbb)):
            probs, _ = predict(model, subset, spec["context"], tokenizer, dev)
            out[name] = breakdown(subset, probs, choice["threshold"], choice["temperature"])
        metrics_file = EXPERIMENTS_DIR / experiment_id / "metrics.json"
        stored = json.loads(metrics_file.read_text())
        stored.update(out)
        metrics_file.write_text(json.dumps(stored, indent=2) + "\n")
        results[experiment_id] = out
        print(f"{experiment_id}: TEST scored", flush=True)
        del model
    return results


def flags(test):
    per = {**test["TEST1"]["per_source"], **test["TEST2"]["per_source"]}
    recalls = {s: per[s]["recall"] for s in ("tensor_trust", "jailbreakllms", "deepset", "gandalf") if s in per}
    fprs = {s: per[s]["fpr"] for s in ("jailbreakllms", "security_docs", "arxiv_abstracts", "deepset") if s in per}
    failure = (min(recalls.values()) < 0.5 and max(recalls.values()) >= 0.8) or max(fprs.values()) > 0.20
    return recalls, fprs, failure


def main():
    lock, rows, tokenizer = select()
    print(f"selection frozen: candidate {lock['dev_selected_candidate']}", flush=True)
    results = test_once(lock, rows, tokenizer)
    v2 = json.loads((ROOT / "reports/v2_public_baseline.json").read_text())["splits"]
    v2_src = {**v2["TEST"], **v2["OOD_TEST"]}
    summary = {}
    for experiment_id, out in results.items():
        if "TEST1" not in out:
            continue
        recalls, fprs, failure = flags(out)
        summary[experiment_id] = {"recall_by_source": recalls, "fpr_by_source": fprs,
                                  "GENERALIZATION_FAILURE": failure,
                                  "calibrated_test_ece": out["TEST1"]["pooled_calibrated"]["ece"]}
    cand = lock["dev_selected_candidate"]
    c = {**results[cand]["TEST1"]["per_source"], **results[cand]["TEST2"]["per_source"]}
    beats = (c["jailbreakllms"]["f1"] > v2_src["jailbreakllms"]["f1"] and c["deepset"]["f1"] > v2_src["deepset"]["f1"]
             and c["tensor_trust"]["recall"] > v2_src["tensor_trust"]["recall"]
             and c["gandalf"]["recall"] > v2_src["gandalf"]["recall"]
             and all(c[s]["fpr"] < v2_src[s]["fpr"] for s in ("jailbreakllms", "security_docs", "arxiv_abstracts", "deepset")))
    s = summary[cand]
    if not beats:
        status = "NO_GO"
    elif s["GENERALIZATION_FAILURE"]:
        status = "EXPERIMENTAL"
    elif max(s["fpr_by_source"].values()) > 0.05 or s["calibrated_test_ece"] > 0.10:
        status = "PROMISING"
    else:
        status = "READY_FOR_EXTERNAL_VALIDATION"
    master = {"test_evaluated": True, "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "lock_sha256": sha256(LOCK), "dev_selected_candidate": cand, "candidate_beats_v2": beats,
              "V5_STATUS": status, "status_rules": STATUS_RULES, "per_experiment": {}}
    for experiment_id in EXPERIMENTS:
        d = EXPERIMENTS_DIR / experiment_id
        if not (d / "selection.json").exists():
            master["per_experiment"][experiment_id] = {"status": "not_trained"}
            continue
        m = json.loads((d / "metrics.json").read_text())
        log = json.loads((d / "training_log.json").read_text())
        master["per_experiment"][experiment_id] = {
            "config": json.loads((d / "config.json").read_text()) | {"selection_rules": "see lock"},
            "selection": json.loads((d / "selection.json").read_text()),
            "checkpoint_sha256": (d / "checkpoint.sha256").read_text().split()[0],
            "parameters": m["parameters"], "checkpoint_bytes": m["checkpoint_bytes"],
            "best_epoch": log["best_epoch"], "wall_clock_training_s": log["wall_clock_training_s"],
            "epochs": log["epochs"], "train_rows": json.loads((d / "dataset_manifest.json").read_text())["train_rows"],
            "DEV": m["DEV_selected_threshold"]["pooled"], "DEV_per_source": m["DEV_selected_threshold"]["per_source"],
            "TEST1": m["TEST1"]["pooled"], "TEST1_per_source": m["TEST1"]["per_source"],
            "TEST2_per_source": m["TEST2"]["per_source"], "flags": summary[experiment_id]}
    MASTER.write_text(json.dumps(master, indent=2) + "\n")
    print(f"V5_STATUS = {status} (candidate {cand}, beats V2: {beats})")


if __name__ == "__main__":
    main()
