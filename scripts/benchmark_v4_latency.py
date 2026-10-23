"""Local Mac latency for the isolated v4 head; not an UNO Q measurement."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from evaluate_v4_windows import score_batches
from run_external_real import jailbreak_samples
from run_v3_external_eval import load_model
from trustlaya.context_windows import read_windows
from trustlaya.utils import normalize

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"benchmarks/external/predictions"
OUT=ROOT/"models/trustlaya-s-v4-research"


def main():
    clean={r["sample_id"] for r in json.loads((LOCAL/"v4_frozen_inputs/v3_jailbreak_predictions.json").read_text())}
    loading=time.perf_counter()
    model,tokenizer,_,_,device=load_model()
    tokenizer.backend_tokenizer.no_truncation();tokenizer.backend_tokenizer.no_padding()
    head=torch.nn.Linear(model.encoder.config.hidden_size,1)
    head.load_state_dict(load_file(OUT/"attack_intent_head.safetensors"));head.eval()
    load_ms=(time.perf_counter()-loading)*1000
    buckets={"<=94":[],"95-510":[],">510":[]}
    for row in jailbreak_samples()[0]:
        if row["sample_id"] not in clean:continue
        content=tokenizer.backend_tokenizer.encode(normalize(row["text"]),add_special_tokens=False).ids
        bucket="<=94" if len(content)<=94 else "95-510" if len(content)<=510 else ">510"
        if len(buckets[bucket])<10:buckets[bucket].append((row["sample_id"],content))
        if all(len(v)==10 for v in buckets.values()):break
    cold_start=time.perf_counter()
    score_batches(model,head,device,tokenizer,[buckets["<=94"][0][1]],batch_size=1)
    if device.type=="mps":torch.mps.synchronize()
    cold_ms=(time.perf_counter()-cold_start)*1000
    observations=[]
    for group,rows in buckets.items():
        for sample_id,content in rows:
            parts=read_windows(content)["WINDOW_MAX"]
            started=time.perf_counter()
            score_batches(model,head,device,tokenizer,parts,batch_size=24)
            if device.type=="mps":torch.mps.synchronize()
            elapsed=(time.perf_counter()-started)*1000
            observations.append({"sample_id":sample_id,"length_bucket":group,"token_length":len(content),
                                 "windows":len(parts),"encoder_batches":(len(parts)+23)//24,"latency_ms":elapsed})
    summary={"device":device.type,"hardware":"local Mac, not Arduino UNO Q", "model_load_ms":load_ms,
             "first_inference_cold_ms":cold_ms,
             "sample_n":len(observations),"window_latency_includes_python_and_encoder_not_policy_or_network":True,
             "by_length":{}}
    for group in buckets:
        selected=[r for r in observations if r["length_bucket"]==group]
        times=[r["latency_ms"] for r in selected]
        summary["by_length"][group]={"n":len(times),"mean_windows":float(np.mean([r["windows"] for r in selected])),
                                      "p50_ms":float(np.percentile(times,50)),"p95_ms":float(np.percentile(times,95)),
                                      "p99_ms":float(np.percentile(times,99))}
    (ROOT/"reports/v4_latency.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    (LOCAL/"v4_latency_rows.json").write_text(json.dumps(observations,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
