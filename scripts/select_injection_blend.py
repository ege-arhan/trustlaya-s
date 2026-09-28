"""Select a two-domain injection-head blend using development splits only."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from train_advanced_pii import embeddings
from train_injection_candidate import DEST as BPI_CANDIDATE, ROOT, SOURCE, REVISION, measures
from trustlaya.dataset import read_rows
from trustlaya.model import BACKBONE, TrustLaya
from trustlaya.utils import device, seed_all


DEST = ROOT / "models/candidates/injection_blend"


def main():
    seed_all(42)
    torch.set_num_threads(4)
    base = ROOT / "models/trustlaya-s-v2"
    source = hf_hub_download(SOURCE, "validation.jsonl", repo_type="dataset", revision=REVISION)
    bpi = [{"text": row["text"], "label": int(row["label"] == "injection")}
           for row in (json.loads(line) for line in open(source))]
    synthetic = [{"text": row["text"], "label": row["labels"]["prompt_injection"]}
                 for row in read_rows(ROOT / "data/splits/val.jsonl")]
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(base / "model.safetensors")
    model.to(device()).eval()
    vectors = {"bpi": embeddings(model, tokenizer, bpi),
               "synthetic": embeddings(model, tokenizer, synthetic)}
    rows = {"bpi": bpi, "synthetic": synthetic}
    states = [load_file(path / "model.safetensors") for path in (base, BPI_CANDIDATE)]
    temperatures = [json.loads((path / "calibration.json").read_text())["prompt_injection"]
                    for path in (base, BPI_CANDIDATE)]
    weights = [state["heads.risks.weight"][2].numpy() / temperature
               for state, temperature in zip(states, temperatures)]
    biases = [float(state["heads.risks.bias"][2]) / temperature
              for state, temperature in zip(states, temperatures)]
    baseline_synthetic = measures(synthetic, vectors["synthetic"] @ weights[0] + biases[0], 1, .5)
    trials = []
    for alpha in (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1):
        weight = (1 - alpha) * weights[0] + alpha * weights[1]
        bias = (1 - alpha) * biases[0] + alpha * biases[1]
        logits = {name: vector @ weight + bias for name, vector in vectors.items()}
        for threshold in np.arange(.25, .801, .05):
            result = {name: measures(rows[name], logits[name], 1, threshold)
                      for name in rows}
            trials.append({"alpha": alpha, "threshold": float(threshold),
                           "bpi": result["bpi"], "synthetic": result["synthetic"]})
    eligible = [row for row in trials
                if row["bpi"]["false_positive_rate"] <= .12
                and row["synthetic"]["false_positive_rate"] <=
                baseline_synthetic["false_positive_rate"] + .05]
    if not eligible:
        least_violation = min(trials, key=lambda row: (
            max(0, row["bpi"]["false_positive_rate"] - .12) +
            max(0, row["synthetic"]["false_positive_rate"] -
                baseline_synthetic["false_positive_rate"] - .05),
            -(row["bpi"]["f1"] + row["synthetic"]["f1"]) / 2))
        report = {"status": "rejected_no_feasible_blend",
                  "constraint": "BPI development FPR <= 0.12 and synthetic development FPR <= baseline + 0.05",
                  "baseline_synthetic": baseline_synthetic,
                  "least_violation": {"alpha": least_violation["alpha"],
                                      "threshold": least_violation["threshold"],
                                      "bpi_f1": least_violation["bpi"]["f1"],
                                      "bpi_fpr": least_violation["bpi"]["false_positive_rate"],
                                      "synthetic_f1": least_violation["synthetic"]["f1"],
                                      "synthetic_fpr": least_violation["synthetic"]["false_positive_rate"]},
                  "all_trials": [{"alpha": row["alpha"], "threshold": row["threshold"],
                                  "bpi_f1": row["bpi"]["f1"],
                                  "bpi_fpr": row["bpi"]["false_positive_rate"],
                                  "synthetic_f1": row["synthetic"]["f1"],
                                  "synthetic_fpr": row["synthetic"]["false_positive_rate"]}
                                 for row in trials]}
        (ROOT / "reports/injection_blend_selection.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "least_violation")}, indent=2))
        return
    selected = max(eligible, key=lambda row: ((row["bpi"]["f1"] + row["synthetic"]["f1"]) / 2,
                                             -row["bpi"]["false_positive_rate"]))
    alpha = selected["alpha"]
    weight = (1 - alpha) * weights[0] + alpha * weights[1]
    bias = (1 - alpha) * biases[0] + alpha * biases[1]
    with torch.no_grad():
        model.heads.risks.weight[2].copy_(torch.tensor(weight, device=device(), dtype=torch.float32))
        model.heads.risks.bias[2].copy_(torch.tensor(bias, device=device(), dtype=torch.float32))
    DEST.mkdir(parents=True, exist_ok=True)
    model.save(DEST / "model.safetensors")
    for file in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        shutil.copy2(base / file, DEST / file)
    calibration = json.loads((base / "calibration.json").read_text())
    calibration["prompt_injection"] = 1.0
    (DEST / "calibration.json").write_text(json.dumps(calibration, indent=2))
    thresholds = json.loads((base / "decision_thresholds.json").read_text())
    thresholds["prompt_injection"] = selected["threshold"]
    (DEST / "decision_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    shutil.copy2(base / "policy.yaml", DEST / "policy.yaml")
    report = {"selection": "Mean BPI/synthetic development F1, BPI FPR <= 0.12, synthetic FPR <= v2 development FPR + 0.05",
              "bpi_revision": REVISION, "bpi_validation_n": len(bpi),
              "synthetic_validation_n": len(synthetic),
              "base_synthetic_validation": baseline_synthetic,
              "eligible_trials": len(eligible), "selected": selected,
              "all_trials": [{"alpha": row["alpha"], "threshold": row["threshold"],
                              "bpi_f1": row["bpi"]["f1"],
                              "bpi_fpr": row["bpi"]["false_positive_rate"],
                              "synthetic_f1": row["synthetic"]["f1"],
                              "synthetic_fpr": row["synthetic"]["false_positive_rate"]}
                             for row in trials],
              "warning": "Only development data selected the blend. BPI labels cover a broader attack task than injection. Do not promote without cross-source regression gates."}
    (ROOT / "reports/injection_blend_selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_alpha": alpha, "threshold": selected["threshold"],
                      "bpi_f1": selected["bpi"]["f1"],
                      "synthetic_f1": selected["synthetic"]["f1"],
                      "bpi_fpr": selected["bpi"]["false_positive_rate"],
                      "synthetic_fpr": selected["synthetic"]["false_positive_rate"]}, indent=2))


if __name__ == "__main__":
    main()
