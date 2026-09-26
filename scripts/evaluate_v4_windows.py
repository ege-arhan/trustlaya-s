"""No-training v3 read-strategy comparison on independent proxy DEV sources."""
from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from run_external_real import metrics
from run_v3_external_eval import load_model, apply_calibration
from trustlaya.context_windows import read_windows
from trustlaya.utils import normalize

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "benchmarks/external/predictions"
THRESHOLD = 0.30  # Already fixed on v3 development, not selected here.


def aggregate(probabilities: list[float]) -> dict[str, float]:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1-1e-7)
    return {"SLIDING": float(p.mean()), "WINDOW_MAX": float(p.max()),
            "WINDOW_LOGIT_POOL": float(1 / (1 + np.exp(-np.log(p/(1-p)).mean())))}


def score_batches(model, head, device, tokenizer, sequences, batch_size=24):
    result = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start+batch_size]
            ids = [tokenizer.build_inputs_with_special_tokens(row) for row in batch]
            mask = [[1] * len(row) + [0] * (96-len(row)) for row in ids]
            ids = [row + [tokenizer.pad_token_id] * (96-len(row)) for row in ids]
            t_ids = torch.tensor(ids, dtype=torch.long, device=device)
            t_mask = torch.tensor(mask, dtype=torch.long, device=device)
            hidden = model.encoder(input_ids=t_ids, attention_mask=t_mask).last_hidden_state
            pooled = (hidden*t_mask.unsqueeze(-1)).sum(1)/t_mask.sum(1, keepdim=True)
            result.extend(torch.sigmoid(head(pooled.cpu())).flatten().tolist())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--head",choices=["v3","v4"],default="v3")
    args=parser.parse_args()
    rows = json.loads((LOCAL/"v4_dev_text_private.json").read_text())
    model, tokenizer, _, head, device = load_model()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    if args.head=="v4":
        head=torch.nn.Linear(model.encoder.config.hidden_size,1)
        head.load_state_dict(load_file(ROOT/"models/trustlaya-s-v4-research/attack_intent_head.safetensors"))
        head.eval()
        calibration=None
        threshold=.5
    else:
        calibration = json.loads((ROOT/"models/trustlaya-s-v3/calibration.json").read_text())["attack"]
        threshold=THRESHOLD
    jobs = []; document = []
    for row in rows:
        content = tokenizer.backend_tokenizer.encode(normalize(row["text"]), add_special_tokens=False).ids
        variants = read_windows(content)
        indexes = {}; local = {}
        for name, parts in variants.items():
            indexes[name] = []
            for part in parts:
                key = tuple(part)
                if key not in local:
                    local[key] = len(jobs)
                    jobs.append(part)
                indexes[name].append(local[key])
        document.append({"id": row["id"], "sha256": row["sha256"], "gold": row["gold"],
                         "source": row["source"], "token_length": len(content), "indexes": indexes})
    print("DEV documents", len(document), "window jobs", len(jobs), flush=True)
    started=time.perf_counter()
    raw = score_batches(model,head,device,tokenizer,jobs)
    total_ms=(time.perf_counter()-started)*1000
    output = []; by_strategy={k:[] for k in document[0]["indexes"]}
    for item in document:
        scores={}
        for name,indexes in item["indexes"].items():
            if name in ("SLIDING","WINDOW_MAX","WINDOW_LOGIT_POOL"):
                p=aggregate([raw[j] for j in indexes])[name]
            else: p=float(raw[indexes[0]])
            cal=float(apply_calibration([p],calibration)[0]) if calibration else p
            scores[name]={"raw":p,"calibrated":cal,"windows":len(indexes)}
            by_strategy[name].append({"gold":item["gold"],"raw_score":cal})
        output.append({k:item[k] for k in ("id","sha256","gold","source","token_length")}|{"scores":scores})
    results={"cohort":"Bordair live game attacks >=500 chars + OWASP 2026 explanatory paragraphs",
             "model":args.head, "threshold":threshold,
             "calibration":"frozen v3 Gandalf/prompts.chat development fit applied after pooling" if calibration else "none; raw 0.5 diagnostic only",
             "batch_size":24,"total_scored_windows_including_strategy_duplicates":len(jobs),
             "total_batched_model_ms":total_ms,
             "strategies":{name:{"metrics":metrics(data,threshold=threshold),
                                  "mean_windows_per_document":float(np.mean([r["scores"][name]["windows"] for r in output])),
                                  "p95_windows_per_document":float(np.percentile([r["scores"][name]["windows"] for r in output],95))}
                           for name,data in by_strategy.items()}}
    (LOCAL/f"v4_window_dev_predictions_{args.head}.json").write_text(json.dumps(output,indent=2)+"\n")
    (ROOT/f"reports/v4_window_dev_metrics_{args.head}.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({k:v["metrics"] for k,v in results["strategies"].items()},indent=2),flush=True)


if __name__=="__main__": main()
