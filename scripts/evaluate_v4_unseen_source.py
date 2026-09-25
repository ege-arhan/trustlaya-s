"""Fixed-protocol test on deepset's previously unused official test partition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

from evaluate_v4_windows import score_batches
from run_external_real import metrics, norm
from run_v3_external_eval import apply_calibration, load_model
from trustlaya.context_windows import read_windows
from trustlaya.inference import Analyzer
from trustlaya.utils import normalize

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"benchmarks/external/predictions"
OUT=ROOT/"models/trustlaya-s-v4-research"
REV="4f61ecb038e9c3fb77e21034b22511b523772cdd"
FILE="data/test-00000-of-00001-701d16158af87368.parquet"


def main():
    path=hf_hub_download("deepset/prompt-injections",FILE,repo_type="dataset",revision=REV)
    source=pq.read_table(path).to_pylist()
    assert len(source)==116
    rows=[{"id":f"deepset:test:{i}","text":str(r["text"]),"gold":int(r["label"])} for i,r in enumerate(source)]
    train=json.loads((LOCAL/"v4_train_text_private.json").read_text())
    dev=json.loads((LOCAL/"v4_dev_text_private.json").read_text())
    known=train+dev
    vectors=HashingVectorizer(analyzer="char",ngram_range=(4,5),n_features=2**18,alternate_sign=False,norm="l2")
    reference=vectors.transform([norm(r["text"]) for r in known])
    candidates=vectors.transform([norm(r["text"]) for r in rows])
    distance,_=NearestNeighbors(n_neighbors=1,metric="cosine",algorithm="brute",n_jobs=-1).fit(reference).kneighbors(candidates)
    excluded=[r["id"] for r,d in zip(rows,distance[:,0]) if 1-float(d)>=.85]
    rows=[r for r,d in zip(rows,distance[:,0]) if 1-float(d)<.85]
    protocol=json.loads((OUT/"operating_point.json").read_text())
    assert protocol["strategy"]=="WINDOW_MAX"
    model,tokenizer,_,v3_head,device=load_model()
    tokenizer.backend_tokenizer.no_truncation();tokenizer.backend_tokenizer.no_padding()
    head=torch.nn.Linear(model.encoder.config.hidden_size,1)
    head.load_state_dict(load_file(OUT/"attack_intent_head.safetensors"));head.eval()
    jobs=[];index=[]
    for row in rows:
        content=tokenizer.backend_tokenizer.encode(normalize(row["text"]),add_special_tokens=False).ids
        parts=read_windows(content)["WINDOW_MAX"]
        index.append((len(jobs),len(jobs)+len(parts)))
        jobs.extend(parts)
    window_scores=score_batches(model,head,device,tokenizer,jobs)
    raw=[max(window_scores[a:b]) for a,b in index]
    cal=apply_calibration(raw,protocol["calibration"])
    predicted=[{"id":r["id"],"sha256":hashlib.sha256(norm(r["text"]).encode()).hexdigest(),
                "gold":r["gold"],"raw_score":float(p),"calibrated_score":float(c),
                "windows":b-a} for r,p,c,(a,b) in zip(rows,raw,cal,index)]
    v3_cal=json.loads((ROOT/"models/trustlaya-s-v3/calibration.json").read_text())["attack"]
    head_jobs=[]
    for row in rows:
        content=tokenizer.backend_tokenizer.encode(normalize(row["text"]),add_special_tokens=False).ids
        head_jobs.append(read_windows(content)["HEAD"][0])
    v3_raw=score_batches(model,v3_head,device,tokenizer,head_jobs)
    v3_score=apply_calibration(v3_raw,v3_cal)
    analyzer=Analyzer("onnx",model_dir=ROOT/"models/trustlaya-s-v2",
                      onnx_path=ROOT/"models/exported/v2/trustlaya_s.onnx")
    v2_score=[analyzer.analyze(r["text"])["raw_scores"]["prompt_injection"] for r in rows]
    (LOCAL/"v4_deepset_unseen_predictions.json").write_text(json.dumps(predicted,indent=2)+"\n")
    result={"dataset":"deepset/prompt-injections", "revision":REV,"split":"official test",
            "note":"Prompt injection labels are not interchangeable with direct jailbreak intent; source card has sparse annotation provenance.",
            "source_test_n":len(source),"near_train_or_dev_excluded":excluded,"clean_n":len(rows),
            "threshold_frozen_on_Bordair_OWASP_DEV":protocol["threshold"],
            "v2":metrics([{"gold":r["gold"],"raw_score":float(p)} for r,p in zip(rows,v2_score)],threshold=.5),
            "v3":metrics([{"gold":r["gold"],"raw_score":float(p)} for r,p in zip(rows,v3_score)],threshold=.3),
            "v4":metrics([{"gold":r["gold"],"raw_score":r["calibrated_score"]} for r in predicted],threshold=protocol["threshold"])}
    (ROOT/"reports/v4_unseen_external_metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
