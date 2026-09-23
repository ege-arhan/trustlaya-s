"""Independent PII and secret diagnostics; save aggregate statistics only."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer

from trustlaya.evidence import extract, PII_TYPES, SECRET_TYPES
from trustlaya.labels import TASKS

ROOT=Path(__file__).resolve().parents[1]
TOKENIZER=AutoTokenizer.from_pretrained(ROOT/'models/student')
TEMPS=json.loads((ROOT/'models/student/calibration.json').read_text())
SESSION=ort.InferenceSession(str(ROOT/'models/exported/trustlaya.onnx'),providers=['CPUExecutionProvider'])
RNG=random.Random(42)


def predict(rows, task, evidence_types):
    ti=TASKS.index(task)
    score=[]; rule=[]
    for at in range(0,len(rows),24):
        chunk=rows[at:at+24]
        x=TOKENIZER([r['text'].replace('I','ı').lower() for r in chunk],max_length=96,truncation=True,padding='max_length',return_tensors='np')
        logits=SESSION.run(None,{k:v for k,v in x.items() if k in ('input_ids','attention_mask')})[0][:,ti]
        probs=1/(1+np.exp(-logits/float(TEMPS[task])))
        score.extend(map(float,probs))
        rule.extend(any(s['type'] in evidence_types for s in extract(r['text'])) for r in chunk)
    return np.asarray(score),np.asarray(rule)


def metric(y, pred):
    y=np.asarray(y,dtype=int); pred=np.asarray(pred,dtype=int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'n':len(y),'positives':int(y.sum()),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'false_positive_rate':float(fp/(fp+tn)) if fp+tn else None,'false_negative_rate':float(fn/(fn+tp)) if fn+tp else None,'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn)}


def sample(rows, n):
    pos=[r for r in rows if r['label']];neg=[r for r in rows if not r['label']]
    RNG.shuffle(pos);RNG.shuffle(neg)
    p=min(len(pos),n//2); q=min(len(neg),n-p)
    out=pos[:p]+neg[:q];RNG.shuffle(out)
    return out


def main():
    p=hf_hub_download('yusuf-said/turkish-privacy-filter-dataset',filename='tr_privacy_tr_curated.jsonl',repo_type='dataset')
    raw=[json.loads(line) for line in open(p)]
    pii_keys={'account_number','private_person','private_phone','private_email','private_address'}
    pii=[{'text':r['text'],'label':int(any(k.split(':')[0] in pii_keys for k in r['spans'])),'kinds':set(k.split(':')[0] for k in r['spans'])} for r in raw]
    pii=sample(pii,2000)
    score,rule=predict(pii,'pii',PII_TYPES)
    out={'turkish_privacy':{'source':'https://huggingface.co/datasets/yusuf-said/turkish-privacy-filter-dataset','scope':'Supported PII categories only; date and URL excluded; independent synthetic/curated source','model_05':metric([r['label'] for r in pii],score>=.5),'rules':metric([r['label'] for r in pii],rule),'hybrid':metric([r['label'] for r in pii],(score>=.5)|rule)}}
    kind_counts={}
    for kind in sorted(pii_keys):
        mask=np.array([kind in r['kinds'] for r in pii])
        kind_counts[kind]={'n':int(mask.sum()),'rule_recall':float(rule[mask].mean()) if mask.any() else None,'hybrid_recall':float(((score>=.5)|rule)[mask].mean()) if mask.any() else None}
    out['turkish_privacy']['by_positive_span_type']=kind_counts

    p=hf_hub_download('Podric/prowl-secrets-corpus',filename='prowlbench.jsonl',repo_type='dataset')
    raw=[json.loads(line) for line in open(p)]
    # Exclude rows marked real. No source text or matched span is saved.
    bench=sample([r for r in raw if r.get('origin') in ('synthetic','augmented')],2000)
    score,rule=predict(bench,'secret',SECRET_TYPES)
    y=[r['label'] for r in bench]
    out['prowl_secrets']={'source':'https://huggingface.co/datasets/Podric/prowl-secrets-corpus','scope':'Independent synthetic/augmented multilingual cases only; CC BY-NC 4.0, no raw rows redistributed','model_05':metric(y,score>=.5),'rules':metric(y,rule),'hybrid':metric(y,(score>=.5)|rule)}
    by={}
    for kind,n in Counter(r['type'] for r in bench).most_common(12):
        ids=[i for i,r in enumerate(bench) if r['type']==kind]
        by[kind]={'n':n,'label_positive_rate':sum(y[i] for i in ids)/n,'hybrid_positive_rate':float(np.mean(((score>=.5)|rule)[ids]))}
    out['prowl_secrets']['by_type']=by
    target=ROOT/'reports/external_privacy_secret.json';target.write_text(json.dumps(out,indent=2))
    print(json.dumps({k:{m:round(v['f1'],3) for m,v in d.items() if isinstance(v,dict) and 'f1' in v} for k,d in out.items()},indent=2))

if __name__=='__main__': main()
