"""Describe seed and batching variance for V5 recipes. No model or threshold is selected here.

All four configurations are scored with the iteration-2 DEV threshold rule (macro
accuracy over (source,label) DEV groups). The TEST sets were already scored in
iteration 2, so these numbers measure run-to-run variation; they are not a new
blind evaluation. A second DEV rule (class-balanced: attack groups and benign
groups weighted equally) is reported as a post-hoc sensitivity analysis of the
deepset recall drop, labeled as such.

Run: .venv/bin/python scripts/evaluate_v5_variance.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_v5_evidence import fit_temperature, metrics, scale  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from train_v5_ablation import BACKBONE, MIXED, TOKENIZER, load_split, predict, tokenize  # noqa: E402
from train_v5_benign_repair import CONTEXT, NEW_SOURCES, load_rows  # noqa: E402
from trustlaya.utils import device  # noqa: E402
from trustlaya.v5_model import V5Classifier  # noqa: E402

CONFIGS = {
    "E6data_random": {42: "models/v5-ablation/E6_mixed_balanced_focal_510", 43: "models/v5-variance/E6data_random_s43",
                      44: "models/v5-variance/E6data_random_s44"},
    "E6data_grouped": {42: "models/v5-benign-repair/R0_control", 43: "models/v5-variance/E6data_grouped_s43",
                       44: "models/v5-variance/E6data_grouped_s44"},
    "A4data_grouped": {42: "models/v5-benign-repair/A4_multilingual_4x", 43: "models/v5-variance/A4data_grouped_s43",
                       44: "models/v5-variance/A4data_grouped_s44"},
    "A4data_random": {42: "models/v5-variance/A4data_random_s42"},
}
GRID = [round(t, 2) for t in np.arange(0.01, 1.0, 0.01)]


def group_rule(rows, probs, balanced):
    accs = defaultdict(list)
    for r, p in zip(rows, probs):
        accs[(r["source"], r["y"])].append(p)
    def score(t):
        per = {g: np.mean([(p >= t) == bool(g[1]) for p in v]) for g, v in accs.items()}
        if not balanced:
            return float(np.mean(list(per.values())))
        attack = np.mean([v for g, v in per.items() if g[1] == 1])
        benign = np.mean([v for g, v in per.items() if g[1] == 0])
        return float((attack + benign) / 2)
    return max(GRID, key=lambda t: (score(t), -abs(t - 0.5)))


def key_metrics(sets, probs, threshold, temperature):
    out = {}
    def src(name, source):
        rows = [r for r in sets[name] if r["source"] == source]
        p = np.asarray([probs[name][r["sample_id"]] for r in rows])
        return metrics([r["y"] for r in rows], p, threshold), rows, p
    for source in ("tensor_trust", "jailbreakllms", "security_docs", "arxiv_abstracts", "dolly", "oasst2", "aya"):
        m, _, _ = src("TEST1", source)
        out[f"{source}_recall" if source in ("tensor_trust", "jailbreakllms") else f"{source}_fpr"] = \
            m["recall"] if source in ("tensor_trust", "jailbreakllms") else m["fpr"]
        if source == "jailbreakllms":
            out["jailbreakllms_fpr"] = m["fpr"]
    d, _, _ = src("TEST2", "deepset")
    out.update(deepset_fpr=d["fpr"], deepset_recall=d["recall"], deepset_pr_auc=d["pr_auc"])
    out["gandalf_recall"] = src("TEST2", "gandalf")[0]["recall"]
    m, rows, p = src("MASSIVE", "massive")
    out["massive_fpr"] = m["fpr"]
    for lang in ("en", "de", "tr"):
        idx = [i for i, r in enumerate(rows) if r["language"] == lang]
        out[f"massive_fpr_{lang}"] = metrics([0] * len(idx), p[idx], threshold)["fpr"]
    rows1 = sets["TEST1"]
    p1 = np.asarray([probs["TEST1"][r["sample_id"]] for r in rows1])
    out["test1_ece_raw"] = metrics([r["y"] for r in rows1], p1)["ece"]
    out["test1_ece_calibrated"] = metrics([r["y"] for r in rows1], scale(p1, temperature))["ece"]
    return out


def main():
    rows = load_rows()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    sets = {"DEV": load_split(rows, "DEV", set(MIXED) | set(NEW_SOURCES)),
            "TEST1": load_split(rows, "TEST", set(MIXED) | set(NEW_SOURCES)),
            "TEST2": load_split(rows, "OOD_TEST", {"deepset", "gandalf"}),
            "MASSIVE": load_split(rows, "OOD_BENIGN_TEST", {"massive"})}
    for subset in sets.values():
        tokenize(tokenizer, subset)
    dev = device()
    report = {"note": __doc__.strip().splitlines()[0], "runs": {}, "configs": {}}
    for config, seeds in CONFIGS.items():
        for seed, folder in seeds.items():
            path = ROOT / folder / "model.safetensors"
            if not path.exists():
                report["runs"][f"{config}_s{seed}"] = {"status": "missing"}
                continue
            model = V5Classifier(BACKBONE, pretrained=False)
            model.load(path)
            model.to(dev)
            probs = {name: dict(zip([r["sample_id"] for r in subset], predict(model, subset, CONTEXT, tokenizer, dev)[0]))
                     for name, subset in sets.items()}
            dev_p = [probs["DEV"][r["sample_id"]] for r in sets["DEV"]]
            threshold = group_rule(sets["DEV"], dev_p, balanced=False)
            balanced = group_rule(sets["DEV"], dev_p, balanced=True)
            temperature = fit_temperature([r["y"] for r in sets["DEV"]], dev_p)
            main_metrics = key_metrics(sets, probs, threshold, temperature)
            post_hoc = key_metrics(sets, probs, balanced, temperature)
            report["runs"][f"{config}_s{seed}"] = {
                "checkpoint_sha256": sha256(path), "threshold_iteration2_rule": threshold,
                "threshold_class_balanced_posthoc": balanced, "temperature": round(temperature, 4),
                "metrics": main_metrics,
                "posthoc_class_balanced": {k: post_hoc[k] for k in ("deepset_recall", "deepset_fpr", "massive_fpr",
                                                                    "jailbreakllms_recall", "jailbreakllms_fpr",
                                                                    "tensor_trust_recall")}}
            print(f"{config}_s{seed} scored", flush=True)
            del model
        runs = [v for k, v in report["runs"].items() if k.startswith(config + "_s") and "metrics" in v]
        if runs:
            keys = runs[0]["metrics"].keys()
            report["configs"][config] = {"seeds": len(runs), **{
                k: {"mean": round(statistics.mean(vals), 4), "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
                    "min": round(min(vals), 4), "max": round(max(vals), 4)}
                for k in keys for vals in [[r["metrics"][k] for r in runs if r["metrics"][k] is not None]] if vals}}
    (ROOT / "reports/v5_variance_study.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
