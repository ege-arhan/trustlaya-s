"""Read-only external benchmarks; never used for training or calibration."""
import json
import random
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from trustlaya.inference import Analyzer
ROOT=Path(__file__).resolve().parents[1]
def metrics(y,p):
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    return {'n':len(y),'positives':sum(y),'accuracy':float(accuracy_score(y,p)),'precision':float(precision),'recall':float(recall),'f1':float(f1),'false_positive_rate':sum(a==0 and b==1 for a,b in zip(y,p))/max(1,sum(a==0 for a in y)),'false_negative_rate':sum(a==1 and b==0 for a,b in zip(y,p))/max(1,sum(a==1 for a in y))}
def main():
    random.seed(42);a=Analyzer();report={}
    ds=load_dataset('deepset/prompt-injections',split='test')
    y=[int(x['label']) for x in ds];p=[int(a.analyze(x['text'])['prompt_injection']>=.5) for x in ds]
    report['deepset_prompt_injection']={'source':'https://huggingface.co/datasets/deepset/prompt-injections','task':'prompt_injection','metrics':metrics(y,p)};print('deepset',report['deepset_prompt_injection']['metrics'],flush=True)
    ds=load_dataset('rogue-security/real-world-benign-use-cases',split='train')
    p=[int(a.analyze(x['text'])['prompt_injection']>=.5) for x in ds]
    report['rogue_real_benign']={'source':'https://huggingface.co/datasets/rogue-security/real-world-benign-use-cases','task':'prompt_injection false positives on hard production benign cases','n':len(p),'false_positives':sum(p),'false_positive_rate':sum(p)/len(p)};print('rogue',report['rogue_real_benign'],flush=True)
    ds=load_dataset('Overfit-GM/turkish-toxic-language',split='train')
    ids=random.sample(range(len(ds)),min(2000,len(ds)));rows=[ds[i] for i in ids]
    y=[int(x['is_toxic']) for x in rows];p=[int(a.analyze(x['text'])['ethics_risk']>=.5) for x in rows]
    report['turkish_toxicity_proxy']={'source':'https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language','task':'ethics_risk as a NON-EQUIVALENT toxicity proxy; diagnostic only','metrics':metrics(y,p),'source_distribution':dict(Counter(x['source'] for x in rows))};print('toxic',report['turkish_toxicity_proxy']['metrics'],flush=True)
    ds=load_dataset('emmanuelgjr/genai-incidents',split='train')
    rows=[x for x in ds if x['quality_tier']=='reviewed' and x['category']=='real-world']
    random.shuffle(rows);rows=rows[:300];scores=[a.analyze((x['title'] or '')+'. '+(x['description'] or '')) for x in rows]
    cats=Counter(x['corpus'] for x in rows)
    report['reviewed_real_incidents']={'source':'https://huggingface.co/datasets/emmanuelgjr/genai-incidents','task':'risk detection on real-world reviewed incidents; positive-only coverage, NOT F1','n':len(rows),'categories':dict(cats),'any_risk_recall_at_05':sum(max(v[k] for k in ('privacy_risk','security_risk','ethics_risk','oversight_risk','data_governance_risk'))>=.5 for v in scores)/max(1,len(rows)),'review_or_block_rate':sum(v['action'] in ('REVIEW','BLOCK','REDACT') for v in scores)/max(1,len(rows))};print('incidents',report['reviewed_real_incidents'],flush=True)
    (ROOT/'reports/external_evaluation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
