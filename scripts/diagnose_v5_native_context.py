"""No-training 94/510-token reading ablation on the old V4 proxy DEV."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from run_external_real import metrics
from run_v3_external_eval import apply_calibration, load_model
from run_v5_context_diagnostic import score_windows
from trustlaya.utils import normalize

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"
STRATEGIES = ("FIRST_94", "NATIVE_FIRST_510", "HEAD_TAIL_510", "SLIDING_256_MAX")


def variants(tokens: list[int]) -> dict[str, list[list[int]]]:
    if len(tokens) <= 510:
        head_tail = tokens
    else:
        head_tail = tokens[:255] + tokens[-255:]
    starts = list(range(0, max(len(tokens) - 256 + 1, 1), 128))
    if starts[-1] != max(0, len(tokens) - 256):
        starts.append(max(0, len(tokens) - 256))
    return {"FIRST_94": [tokens[:94]], "NATIVE_FIRST_510": [tokens[:510]],
            "HEAD_TAIL_510": [head_tail],
            "SLIDING_256_MAX": [tokens[start:start + 256] for start in starts]}


def main() -> None:
    rows = json.loads((ROOT / "benchmarks/external/predictions/v4_dev_text_private.json").read_text())
    model, tokenizer, _, v3_head, device = load_model()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    v4_head = torch.nn.Linear(model.encoder.config.hidden_size, 1)
    v4_head.load_state_dict(load_file(ROOT / "models/trustlaya-s-v4-research/attack_intent_head.safetensors"))
    v4_head.eval()
    heads = {"v3": v3_head, "v4": v4_head}
    all_windows = []
    plans = []
    for row in rows:
        tokens = tokenizer.backend_tokenizer.encode(normalize(row["text"]), add_special_tokens=False).ids
        mapping, local = {}, {}
        for strategy, windows in variants(tokens).items():
            mapping[strategy] = []
            for window in windows:
                key = tuple(window)
                if key not in local:
                    local[key] = len(all_windows)
                    all_windows.append(window)
                mapping[strategy].append(local[key])
        plans.append({"id": row["id"], "gold": row["gold"], "source": row["source"],
                      "length": len(tokens), "indexes": mapping})
    print(f"Old proxy DEV {len(rows)} rows; {len(all_windows)} unique windows", flush=True)
    raw = score_windows(model, heads, device, tokenizer, all_windows)
    v3_fit = json.loads((ROOT / "models/trustlaya-s-v3/calibration.json").read_text())["attack"]
    v4_protocol = json.loads((ROOT / "models/trustlaya-s-v4-research/operating_point.json").read_text())
    fits = {"v3": v3_fit, "v4": v4_protocol["calibration"]}
    thresholds = {"v2": .5, "v3": .3, "v4": v4_protocol["threshold"]}
    grouped = {version: defaultdict(list) for version in thresholds}
    predictions = []
    for plan in plans:
        scores = {}
        for version in thresholds:
            scores[version] = {}
            for strategy, indexes in plan["indexes"].items():
                selected = [raw[version][index] for index in indexes]
                pooled = max(selected) if strategy == "SLIDING_256_MAX" else selected[0]
                value = float(apply_calibration([pooled], fits[version])[0]) if version in fits else float(pooled)
                scores[version][strategy] = value
                grouped[version][strategy].append({"gold": plan["gold"], "raw_score": value})
        predictions.append({"sample_id": plan["id"], "token_length": plan["length"], "scores": scores})
    latency = defaultdict(list)
    selected_plans = sorted(plans, key=lambda plan: plan["length"])
    sample_indexes = sorted(set(np.linspace(0, len(plans)-1, 40).astype(int).tolist()))
    for index in sample_indexes:
        plan = selected_plans[index]
        for strategy in STRATEGIES:
            windows = [all_windows[j] for j in plan["indexes"][strategy]]
            start = time.perf_counter()
            score_windows(model, heads, device, tokenizer, windows)
            latency[strategy].append((time.perf_counter()-start)*1000)
    output = {"status": "EXPLORATORY_OLD_PROXY_DEV_ONLY", "n": len(plans), "latency_n": len(sample_indexes),
              "device": device.type, "hardware": "local Mac, not UNO Q",
              "label_limit": "Source-derived Bordair game attempts and OWASP docs; not human intent gold",
              "calibration_note": "Frozen old fits applied after pooling; not refit for context strategies",
              "by_model": {version: {strategy: {
                  "metrics": metrics(grouped[version][strategy], threshold=thresholds[version]),
                  "mean_windows_per_input": float(np.mean([len(plan["indexes"][strategy]) for plan in plans])),
                  "p50_latency_ms": float(np.percentile(latency[strategy], 50)),
                  "p95_latency_ms": float(np.percentile(latency[strategy], 95))}
                  for strategy in STRATEGIES} for version in thresholds}}
    (PRIVATE / "old_proxy_native_context_predictions.json").write_text(json.dumps(predictions) + "\n")
    (ROOT / "benchmarks/v5/old_proxy_native_context_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({version: {name: {k: round(data["metrics"][k], 3) for k in ("recall", "fpr", "f1")}
                                for name, data in strategies.items()}
                      for version, strategies in output["by_model"].items()}, indent=2))


if __name__ == "__main__":
    main()
