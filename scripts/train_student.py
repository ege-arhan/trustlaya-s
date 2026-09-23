import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer
from trustlaya.dataset import read_rows
from trustlaya.labels import TASKS,ACTIONS,SEVERITIES
from trustlaya.model import TrustLaya,BACKBONE
from trustlaya.distillation import loss
from trustlaya.utils import seed_all,device,normalize
ROOT=Path(__file__).resolve().parents[1]
class Rows(Dataset):
    def __init__(self,rows,tokenizer,teacher=None):
        self.rows=rows; self.teacher=teacher or {}; self.enc=tokenizer([normalize(r["text"]) for r in rows],truncation=True,max_length=96,padding="max_length",return_tensors="pt")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return self.enc["input_ids"][i],self.enc["attention_mask"][i],torch.tensor([r["labels"][k] for k in TASKS],dtype=torch.float32),torch.tensor(SEVERITIES.index(r["severity"])),torch.tensor(ACTIONS.index(r["action"])),torch.tensor([self.teacher.get(r["text"],{}).get(k,float("nan")) for k in TASKS],dtype=torch.float32)
def main():
    p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=200);p.add_argument("--batch",type=int,default=32);p.add_argument("--lr",type=float,default=3e-4);p.add_argument("--resume",action="store_true");p.add_argument("--teacher-file");a=p.parse_args()
    seed_all(); torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(BACKBONE)
    rows=read_rows(ROOT/"data/splits/train.jsonl")
    teacher={r["text"]:r["teacher"] for r in read_rows(a.teacher_file)} if a.teacher_file else {}
    data=Rows(rows,tokenizer,teacher); loader=DataLoader(data,batch_size=a.batch,shuffle=True)
    model=TrustLaya(BACKBONE,pretrained=not a.resume)
    if a.resume: model.load(ROOT/"models/student/model.safetensors")
    model=model.to(device()); model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p=0.0
        if hasattr(module, "dropout_prob"): module.dropout_prob=0.0
    # Freeze lower encoder layers to keep overnight training bounded.
    for name,param in model.encoder.named_parameters():
        if name.startswith("embeddings") or name.startswith("encoder.layer.0") or name.startswith("encoder.layer.1"):
            param.requires_grad=False
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr)
    step=0; losses=[]
    while step<a.steps:
        for batch in loader:
            ids,mask,risk,sev,act,soft=[x.to(device()) for x in batch]
            opt.zero_grad(set_to_none=True)
            result=model(ids,mask); value=loss(result,risk,sev,act,soft if teacher else None)
            value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            losses.append(float(value.item()));step+=1
            if step%25==0:print(f"step={step} loss={sum(losses[-25:])/25:.4f}",flush=True)
            if step>=a.steps:break
    target=ROOT/"models/student";target.mkdir(parents=True,exist_ok=True)
    model.save(target/"model.safetensors");tokenizer.save_pretrained(target)
    report={"backbone":BACKBONE,"steps":step,"training_examples_seen":step*a.batch,"last_loss":losses[-1],"parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"teacher_used":bool(teacher),"teacher_examples":len(teacher)}
    (ROOT/"reports/training.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
