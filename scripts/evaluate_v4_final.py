"""One fixed v4 read on the frozen, untouched 5,761-row JLL external cohort."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from evaluate_v4_windows import score_batches
from run_external_real import jailbreak_samples, metrics
from run_v3_external_eval import apply_calibration, load_model
from trustlaya.context_windows import read_windows
from trustlaya.v3_external import fingerprint
from trustlaya.utils import normalize

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"benchmarks/external/predictions"
OUT=ROOT/"models/trustlaya-s-v4-research"


def main():
    protocol=json.loads((OUT/"operating_point.json").read_text())
    assert protocol["strategy"]=="WINDOW_MAX"
    manifest=json.loads((OUT/"manifest.json").read_text())
    assert hashlib.sha256((OUT/"attack_intent_head.safetensors").read_bytes()).hexdigest()==manifest["head_sha256"]
    frozen_v2=json.loads((LOCAL/"v4_frozen_inputs/v2_jailbreak_predictions.json").read_text())
    frozen_v3=json.loads((LOCAL/"v4_frozen_inputs/v3_jailbreak_predictions.json").read_text())
    v2_map={r["sample_id"]:r for r in frozen_v2}
    v3_map={r["sample_id"]:r for r in frozen_v3}
    model,tokenizer,_,_,device=load_model()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    head=torch.nn.Linear(model.encoder.config.hidden_size,1)
    head.load_state_dict(load_file(OUT/"attack_intent_head.safetensors"));head.eval()
    samples=[r for r in jailbreak_samples()[0] if r["sample_id"] in v3_map]
    assert len(samples)==len(v3_map)==5761
    assert all(fingerprint(s["text"])==v3_map[s["sample_id"]]["text_hash"] and s["gold"]==v3_map[s["sample_id"]]["gold"] for s in samples)
    windows=[]; lengths=[]; offsets=[]
    for sample in samples:
        tokens=tokenizer.backend_tokenizer.encode(normalize(sample["text"]),add_special_tokens=False).ids
        parts=read_windows(tokens)["WINDOW_MAX"]
        offsets.append((len(windows),len(windows)+len(parts)))
        windows.extend(parts);lengths.append(len(tokens))
    print("final frozen rows",len(samples),"WINDOW_MAX jobs",len(windows),flush=True)
    window_scores=score_batches(model,head,device,tokenizer,windows,batch_size=24)
    raw=[max(window_scores[a:b]) for a,b in offsets]
    scores=apply_calibration(raw,protocol["calibration"])
    output=[]
    for sample,length,prob,cal in zip(samples,lengths,raw,scores,strict=True):
        output.append({"sample_id":sample["sample_id"],"source":sample["source"],"gold":sample["gold"],
                       "text_hash":fingerprint(sample["text"]),"token_length":length,
                       "raw_score":float(prob),"calibrated_score":float(cal),
                       "predicted_label":int(cal>=protocol["threshold"])})
    (LOCAL/"v4_jailbreak_final_predictions.json").write_text(json.dumps(output,indent=2)+"\n")
    v2_rows=[{"gold":r["gold"],"raw_score":v2_map[r["sample_id"]]["raw_score"]} for r in output]
    v3_rows=[{"gold":r["gold"],"raw_score":v3_map[r["sample_id"]]["calibrated_score"]} for r in output]
    result={"dataset":"JailbreakLLMs clean external","n":len(output),
            "test_source":"community Reddit/Discord/website; previously examined for v2/v3, not used for v4 fit/selection",
            "v2":{"threshold":.5,"metrics":metrics(v2_rows,threshold=.5)},
            "v3":{"threshold":.3,"metrics":metrics(v3_rows,threshold=.3)},
            "v4":{"threshold":protocol["threshold"],"strategy":protocol["strategy"],
                  "calibration":protocol["calibration"],"metrics":metrics([{"gold":r["gold"],"raw_score":r["calibrated_score"]} for r in output],threshold=protocol["threshold"])},
            "v4_by_source":{},"v4_by_length":{}}
    for source in sorted({r["source"] for r in output}):
        rows=[{"gold":r["gold"],"raw_score":r["calibrated_score"]} for r in output if r["source"]==source]
        result["v4_by_source"][source]=metrics(rows,threshold=protocol["threshold"])
    for name,predicate in (("<=94",lambda n:n<=94),("95-510",lambda n:94<n<=510),(">510",lambda n:n>510)):
        rows=[{"gold":r["gold"],"raw_score":r["calibrated_score"]} for r in output if predicate(r["token_length"])]
        result["v4_by_length"][name]=metrics(rows,threshold=protocol["threshold"]) if rows else {"n":0}
    (ROOT/"reports/v4_final_external_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v["metrics"] for k,v in result.items() if k in ("v2","v3","v4")},indent=2),flush=True)


if __name__=="__main__":main()
