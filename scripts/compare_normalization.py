import json
import re
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import trustlaya.inference as inference
from trustlaya.dataset import read_rows
ROOT=Path(__file__).resolve().parents[1]
a=inference.Analyzer()
synthetic=read_rows(ROOT/'data/splits/val.jsonl')
external=load_dataset('deepset/prompt-injections',split='train')
y=np.array([int(x['label']) for x in external]);_,idx=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
sets={'synthetic':[(x['text'],x['labels']['prompt_injection']) for x in synthetic],'external_internal_val':[(external[int(i)]['text'],int(y[i])) for i in idx]}
old=inference.normalize
variants={'original':old,'language_aware':lambda t:(t.replace('I','ı').replace('İ','i').lower() if re.search('[çğıöşüÇĞİÖŞÜ]',t) else t.lower())}
report={}
for name,normalizer in variants.items():
    inference.normalize=normalizer;report[name]={}
    for split,rows in sets.items():
        truth=[y for _,y in rows];pred=[int(a.analyze(text)['prompt_injection']>=.5) for text,_ in rows]
        report[name][split]={'f1':f1_score(truth,pred),'false_positive_rate':sum(t==0 and p==1 for t,p in zip(truth,pred))/max(1,sum(t==0 for t in truth))}
(ROOT/'reports/normalization_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
