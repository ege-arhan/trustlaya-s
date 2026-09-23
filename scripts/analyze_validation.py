import json
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score,recall_score
from trustlaya.dataset import read_rows
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS
ROOT=Path(__file__).resolve().parents[1]
rows=read_rows(ROOT/'data/splits/val.jsonl')
a=Analyzer()
pred=[a.analyze(r['text']) for r in rows]
report={'examples':len(rows),'tasks':{},'by_language':{},'errors':{}}
for task in TASKS:
    y=[r['labels'][task] for r in rows];p=[int(x[task]>=.5) for x in pred]
    report['tasks'][task]={'f1':f1_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'support':sum(y)}
    report['errors'][task]=[{'text':r['text'],'truth':r['labels'][task],'probability':round(x[task],3)} for r,x in zip(rows,pred) if int(x[task]>=.5)!=r['labels'][task]][:12]
for lang in sorted({r['language'] for r in rows}):
    ids=[i for i,r in enumerate(rows) if r['language']==lang]
    report['by_language'][lang]={task:f1_score([rows[i]['labels'][task] for i in ids],[int(pred[i][task]>=.5) for i in ids],zero_division=0) for task in TASKS}
report['macro_f1']=sum(v['f1'] for v in report['tasks'].values())/len(TASKS)
(ROOT/'reports/validation_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps({'macro_f1':report['macro_f1'],'tasks':report['tasks'],'by_language':report['by_language']},indent=2))
