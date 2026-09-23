"""Train only injection head on independent deepset train; never reads its test split."""
import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from trustlaya.model import TrustLaya,BACKBONE
from trustlaya.utils import normalize,seed_all,device
ROOT=Path(__file__).resolve().parents[1]
def main():
    seed_all();torch.set_num_threads(4)
    ds=load_dataset('deepset/prompt-injections',split='train')
    texts=[x['text'] for x in ds];labels=np.array([int(x['label']) for x in ds])
    train_idx,val_idx=train_test_split(np.arange(len(ds)),test_size=.2,random_state=42,stratify=labels)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/student')
    enc=tok([normalize(x) for x in texts],padding='max_length',truncation=True,max_length=96,return_tensors='pt')
    model=TrustLaya(BACKBONE,False);model.load(ROOT/'models/student/model.safetensors');model.to(device())
    for p in model.parameters():p.requires_grad=False
    model.heads.risks.weight.requires_grad=True;model.heads.risks.bias.requires_grad=True
    opt=torch.optim.AdamW([model.heads.risks.weight,model.heads.risks.bias],lr=.003,weight_decay=0)
    best=-1;best_state=None;history=[]
    for epoch in range(1,11):
        model.eval();rng=np.random.default_rng(42+epoch);indices=rng.permutation(train_idx)
        for start in range(0,len(indices),32):
            ix=indices[start:start+32]
            ids=enc['input_ids'][ix].to(device());mask=enc['attention_mask'][ix].to(device());y=torch.tensor(labels[ix],dtype=torch.float32,device=device())
            opt.zero_grad(set_to_none=True);logits=model(ids,mask)[0][:,2]
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y);loss.backward();opt.step()
        model.eval();probs=[]
        with torch.inference_mode():
            for start in range(0,len(val_idx),32):
                ix=val_idx[start:start+32]
                p=torch.sigmoid(model(enc['input_ids'][ix].to(device()),enc['attention_mask'][ix].to(device()))[0][:,2])
                probs.extend(p.cpu().tolist())
        f1=f1_score(labels[val_idx],[int(p>=.5) for p in probs]);history.append({'epoch':epoch,'validation_f1':f1})
        print('epoch',epoch,'validation_f1',round(f1,4),flush=True)
        if f1>best:
            best=f1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    dest=ROOT/'models/candidates/injection_real';dest.mkdir(parents=True,exist_ok=True)
    model.save(dest/'model.safetensors')
    (ROOT/'reports/injection_finetune.json').write_text(json.dumps({'source':'deepset/prompt-injections train','train_examples':len(train_idx),'validation_examples':len(val_idx),'best_internal_validation_f1':best,'history':history},indent=2))
    print('saved',dest)
if __name__=='__main__':main()
