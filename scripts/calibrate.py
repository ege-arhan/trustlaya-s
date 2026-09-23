import json
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
    rows=read_rows(ROOT/"data/splits/val.jsonl");tok=AutoTokenizer.from_pretrained(ROOT/"models/student")
    model=TrustLaya(BACKBONE,False);model.load(ROOT/"models/student/model.safetensors");model.to(device()).eval()
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
    (ROOT/"models/student/calibration.json").write_text(json.dumps(temps,indent=2))
    (ROOT/"reports/calibration.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
