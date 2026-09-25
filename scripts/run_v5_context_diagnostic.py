"""No-training eight-strategy diagnostic on the *old proxy DEV only*.

This is not the new human-reviewed V5 benchmark and cannot select a V5 model.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from run_external_real import metrics
from run_v3_external_eval import apply_calibration, load_model
from trustlaya.utils import normalize
from trustlaya.v5_context import STRATEGIES, aggregate_score, candidate_windows, cue_token_positions

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
V4 = ROOT / "models/trustlaya-s-v4-research"


def score_windows(model, heads, device, tokenizer, windows):
    result = {name: [] for name in ("v2", "v3", "v4")}
    with torch.inference_mode():
        for start in range(0, len(windows), 24):
            batch = windows[start:start + 24]
            ids = [tokenizer.build_inputs_with_special_tokens(row) for row in batch]
            mask = [[1] * len(row) + [0] * (96 - len(row)) for row in ids]
            ids = [row + [tokenizer.pad_token_id] * (96 - len(row)) for row in ids]
            t_ids = torch.tensor(ids, dtype=torch.long, device=device)
            t_mask = torch.tensor(mask, dtype=torch.long, device=device)
            hidden = model.encoder(input_ids=t_ids, attention_mask=t_mask).last_hidden_state
            pooled = (hidden * t_mask.unsqueeze(-1)).sum(1) / t_mask.sum(1, keepdim=True)
            result["v2"].extend(torch.sigmoid(model.heads.risks(pooled)[:, 2]).cpu().tolist())
            cpu = pooled.cpu()
            for version in ("v3", "v4"):
                result[version].extend(torch.sigmoid(heads[version](cpu)).flatten().tolist())
    if device.type == "mps":
        torch.mps.synchronize()
    return result


def main() -> None:
    # This file was used in V4 development and has source/context labels only.
    rows = json.loads((ROOT / "benchmarks/external/predictions/v4_dev_text_private.json").read_text())
    model, tokenizer, _, v3_head, device = load_model()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    v4_head = torch.nn.Linear(model.encoder.config.hidden_size, 1)
    v4_head.load_state_dict(load_file(V4 / "attack_intent_head.safetensors"))
    v4_head.eval()
    heads = {"v3": v3_head, "v4": v4_head}
    fits = {"v3": json.loads((ROOT / "models/trustlaya-s-v3/calibration.json").read_text())["attack"],
            "v4": json.loads((V4 / "operating_point.json").read_text())["calibration"]}
    thresholds = {"v2": 0.5, "v3": 0.3, "v4": 0.79}
    all_windows = []
    plans = []
    for row in rows:
        text = normalize(row["text"])
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        tokens, offsets = encoded["input_ids"], encoded["offset_mapping"]
        cues = cue_token_positions(text, offsets)
        local = {}
        indexes = {}
        for strategy in STRATEGIES:
            indexes[strategy] = []
            for window in candidate_windows(tokens, strategy, cues=cues):
                key = tuple(window)
                if key not in local:
                    local[key] = len(all_windows)
                    all_windows.append(window)
                indexes[strategy].append(local[key])
        plans.append({"sample_id": row["id"], "source": row["source"], "gold": row["gold"],
                      "token_length": len(tokens), "indexes": indexes})
    print(f"Scoring {len(rows)} old proxy DEV rows and {len(all_windows)} unique windows", flush=True)
    raw = score_windows(model, heads, device, tokenizer, all_windows)
    predictions = []
    grouped = {version: defaultdict(list) for version in thresholds}
    for plan in plans:
        scores = {}
        for version in thresholds:
            per_strategy = {}
            for strategy, indexes in plan["indexes"].items():
                pooled = aggregate_score([raw[version][i] for i in indexes], strategy)
                value = float(apply_calibration([pooled], fits[version])[0]) if version in fits else pooled
                per_strategy[strategy] = value
                grouped[version][strategy].append({"gold": plan["gold"], "raw_score": value})
            scores[version] = per_strategy
        predictions.append({"sample_id": plan["sample_id"], "source": plan["source"],
                            "gold": plan["gold"], "token_length": plan["token_length"], "scores": scores})
    latency = defaultdict(list)
    # Deterministic sample across lengths. Latency is measured per strategy,
    # jointly scoring the three heads after one shared encoder pass.
    benchmark_plans = sorted(plans, key=lambda p: p["token_length"])
    sample_indexes = sorted(set(np.linspace(0, len(benchmark_plans) - 1, 40).astype(int).tolist()))
    for index in sample_indexes:
        plan = benchmark_plans[index]
        for strategy in STRATEGIES:
            windows = [all_windows[i] for i in plan["indexes"][strategy]]
            started = time.perf_counter()
            score_windows(model, heads, device, tokenizer, windows)
            latency[strategy].append((time.perf_counter() - started) * 1000)
    output = {"status": "EXPLORATORY_OLD_PROXY_DEV_ONLY", "n": len(rows), "latency_n": len(sample_indexes),
              "device": str(device), "hardware": "local Mac, not UNO Q",
              "label_limit": "Bordair game attack context vs OWASP document source; not human-reviewed per-row intent",
              "calibration_note": "original v3/v4 fits applied after aggregation; not recalibrated per strategy",
              "by_model": {version: {strategy: {
                  "metrics": metrics(grouped[version][strategy], threshold=thresholds[version]),
                  "mean_windows_per_input": float(np.mean([len(p["indexes"][strategy]) for p in plans])),
                  "p50_latency_ms": float(np.percentile(latency[strategy], 50)),
                  "p95_latency_ms": float(np.percentile(latency[strategy], 95)),
              } for strategy in STRATEGIES} for version in thresholds}}
    (PRIVATE / "old_proxy_context_predictions.json").write_text(json.dumps(predictions) + "\n")
    (ROOT / "benchmarks/v5/old_proxy_context_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({version: {name: {"f1": round(stats["metrics"]["f1"], 3),
                                             "recall": round(stats["metrics"]["recall"], 3),
                                             "fpr": round(stats["metrics"]["fpr"], 3)}
                                for name, stats in data.items()} for version, data in output["by_model"].items()}, indent=2))


if __name__ == "__main__":
    main()
