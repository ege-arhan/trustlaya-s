import json
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from trustlaya.dataset import read_rows
from trustlaya.model import TrustLaya,BACKBONE
from trustlaya.labels import TASKS
from trustlaya.calibration import fit_temperature,apply,metrics
from trustlaya.utils import device,normalize
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,default=ROOT/"models/trustlaya-s-v2")
    p.add_argument("--output",type=Path,default=ROOT/"models/candidates/recalibration/calibration.json")
    a=p.parse_args()
    protected={ROOT/"models/student/calibration.json",ROOT/"models/trustlaya-s-v1/calibration.json",ROOT/"models/trustlaya-s-v2/calibration.json"}
    if a.output.resolve() in {path.resolve() for path in protected}:
        p.error("Refusing to overwrite released or evaluated calibration")
    rows=read_rows(ROOT/"data/splits/val.jsonl");tok=AutoTokenizer.from_pretrained(a.model_dir)
    model=TrustLaya(BACKBONE,False);model.load(a.model_dir/"model.safetensors");model.to(device()).eval()
    logits=[]
    for i in range(0,len(rows),32):
        x=tok([normalize(r["text"]) for r in rows[i:i+32]],padding="max_length",truncation=True,max_length=96,return_tensors="pt")
        with torch.inference_mode(): out=model(x["input_ids"].to(device()),x["attention_mask"].to(device()))[0]
        logits.append(out.cpu().numpy())
    logits=np.concatenate(logits);temps={};report={}
    for j,key in enumerate(TASKS):
        y=np.array([r["labels"][key] for r in rows]);t=fit_temperature(y,logits[:,j]);temps[key]=t
        raw=1/(1+np.exp(-logits[:,j]));cal=apply(raw,t)
        report[key]={"raw":metrics(y,raw),"calibrated":metrics(y,cal),"temperature":t}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(temps,indent=2))
    (ROOT/"reports/recalibration_candidate.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
