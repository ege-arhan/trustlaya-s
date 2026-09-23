import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from scipy.special import logit,expit
from sklearn.metrics import f1_score,recall_score
from sklearn.model_selection import train_test_split
from trustlaya.inference import Analyzer
from trustlaya.dataset import read_rows
ROOT=Path(__file__).resolve().parents[1]
base=Analyzer();candidate=Analyzer(model_dir=ROOT/'models/candidates/injection_real')
synthetic=read_rows(ROOT/'data/splits/val.jsonl')
external=load_dataset('deepset/prompt-injections',split='train')
y_ext=np.array([int(x['label']) for x in external]);_,idx=train_test_split(np.arange(len(external)),test_size=.2,random_state=42,stratify=y_ext)
sets={'synthetic':([(x['text'],x['labels']['prompt_injection']) for x in synthetic]),'external_internal_val':[(external[int(i)]['text'],int(y_ext[i])) for i in idx]}
probs={}
for name,rows in sets.items():
    p0=np.array([base.analyze(text)['prompt_injection'] for text,_ in rows]);p1=np.array([candidate.analyze(text)['prompt_injection'] for text,_ in rows]);y=np.array([label for _,label in rows]);probs[name]=(p0,p1,y)
results=[]
for alpha in [0,.1,.2,.3,.4,.5,.6,.8,1]:
    metrics={}
    for name,(p0,p1,y) in probs.items():
        mixed=expit((1-alpha)*logit(np.clip(p0,1e-7,1-1e-7))+alpha*logit(np.clip(p1,1e-7,1-1e-7)))
        pred=(mixed>=.5).astype(int)
        metrics[name]={'f1':float(f1_score(y,pred)),'recall':float(recall_score(y,pred)),'false_positive_rate':float(sum((y==0)&(pred==1))/max(1,sum(y==0)))}
    results.append({'alpha':alpha,**metrics})
valid=[r for r in results if r['synthetic']['f1']>=.95]
best=max(valid,key=lambda r:r['external_internal_val']['f1'])
report={'selection_rule':'maximize independent train-internal validation F1 with synthetic validation F1 >=0.95','results':results,'selected':best}
(ROOT/'reports/injection_blend_selection.json').write_text(json.dumps(report,indent=2))
for r in results:print(r['alpha'],round(r['synthetic']['f1'],3),round(r['external_internal_val']['f1'],3),round(r['external_internal_val']['false_positive_rate'],3))
print('selected',best['alpha'])
