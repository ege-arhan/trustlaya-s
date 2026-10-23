"""Fit an isolated linear v4 research head on a frozen encoder; never reads DEV/TEST."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.linear_model import LogisticRegression

from run_v3_external_eval import load_model
from train_v3_external import encode

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"benchmarks/external/predictions"
OUT=ROOT/"models/trustlaya-s-v4-research"


def main():
    rows=json.loads((LOCAL/"v4_train_text_private.json").read_text())
    y=np.array([r["gold"] for r in rows],dtype=int)
    assert y.sum() >= 50 and len(y)-y.sum() >= 500
    model,tokenizer,_,_,device=load_model()
    features=encode(model,tokenizer,[r["text"] for r in rows],device,False,batch_size=24).astype(np.float32)
    clf=LogisticRegression(C=0.1,class_weight="balanced",max_iter=500,random_state=20260925)
    clf.fit(features,y)
    OUT.mkdir(parents=True,exist_ok=True)
    head=OUT/"attack_intent_head.safetensors"
    save_file({"weight":torch.from_numpy(clf.coef_.astype(np.float32)),
               "bias":torch.from_numpy(clf.intercept_.astype(np.float32))},str(head))
    manifest={"status":"research candidate, not deployed", "version":"v4-linear-head-1",
              "encoder":"frozen v2/v3 shared encoder", "max_length_total":96,
              "training_rows":len(rows),"positive":int(y.sum()),"negative":int(len(y)-y.sum()),
              "optimizer":f"scikit-learn LogisticRegression solver={clf.solver}", "C":0.1,
              "class_weight":"balanced", "random_state":20260925,
              "v2_weight_sha256":hashlib.sha256((ROOT/"models/trustlaya-s-v2/model.safetensors").read_bytes()).hexdigest(),
              "tokenizer_sha256":hashlib.sha256((ROOT/"models/trustlaya-s-v2/tokenizer.json").read_bytes()).hexdigest(),
              "head_sha256":hashlib.sha256(head.read_bytes()).hexdigest(),
              "source_audit":"reports/v4_training_data_audit.json"}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":main()
