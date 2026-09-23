import json
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix
from trustlaya.inference import Analyzer
ROOT=Path(__file__).resolve().parents[1]
a=Analyzer();out={}
for split in ('train','test'):
    ds=load_dataset('deepset/prompt-injections',split=split)
    out[split]={'y':[int(x['label']) for x in ds],'score':[a.analyze(x['text'])['prompt_injection'] for x in ds]}
curve=[]
for threshold in [i/100 for i in range(5,100,5)]:
    y=out['train']['y'];p=[int(s>=threshold) for s in out['train']['score']]
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    curve.append({'threshold':threshold,'f1':f1_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'precision':precision_score(y,p,zero_division=0),'fpr':fp/max(1,fp+tn)})
best=max(curve,key=lambda r:(r['f1'],-r['fpr']))
for threshold in (0.5,0.7,best['threshold']):
    y=out['test']['y'];p=[int(s>=threshold) for s in out['test']['score']]
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    print({'threshold':threshold,'test_f1':f1_score(y,p,zero_division=0),'test_recall':recall_score(y,p,zero_division=0),'test_fpr':fp/max(1,fp+tn)},flush=True)
report={'selection_data':'deepset train only','best_train_f1':best,'train_curve':curve}
(ROOT/'reports/injection_threshold_tuning.json').write_text(json.dumps(report,indent=2))
