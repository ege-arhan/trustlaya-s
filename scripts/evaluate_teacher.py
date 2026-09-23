import json
import time
from pathlib import Path
from sklearn.metrics import f1_score,recall_score
from trustlaya.dataset import read_rows
from trustlaya.labels import TASKS
from trustlaya.evidence import extract,PII_TYPES,SECRET_TYPES
from trustlaya.inference import Analyzer
from trustlaya.calibration import ece
from teacher_inference import QUESTIONS
ROOT=Path(__file__).resolve().parents[1]
def task_metrics(rows,predictions):
    tasks={}
    for k in TASKS:
        y=[r['labels'][k] for r in rows];p=[r[k] for r in predictions];binary=[int(x>=.5) for x in p]
        tasks[k]={'f1':float(f1_score(y,binary,zero_division=0)),'recall':float(recall_score(y,binary,zero_division=0)),'ece':ece(y,p)}
    return {'tasks':tasks,'macro_f1':sum(v['f1'] for v in tasks.values())/len(TASKS),'macro_recall':sum(v['recall'] for v in tasks.values())/len(TASKS),'mean_ece':sum(v['ece'] for v in tasks.values())/len(TASKS)}
def main():
    import laya
    rows=read_rows(ROOT/'data/splits/test.jsonl')[:128]
    teacher=laya.load(str(ROOT/'models/teacher'),device='cpu');student=Analyzer()
    teacher_out=[];teacher_lat=[];student_out=[];student_lat=[];rules=[]
    for row in rows:
        t=time.perf_counter();r=teacher.predict({'text':row['text']},QUESTIONS);teacher_lat.append((time.perf_counter()-t)*1000)
        teacher_out.append({k:r['answers'][k]['noul'] for k in TASKS})
        t=time.perf_counter();student_out.append(student.analyze(row['text']));student_lat.append((time.perf_counter()-t)*1000)
        spans=extract(row['text']);rules.append({k:float(any(x['type'] in (PII_TYPES if k=='pii' else SECRET_TYPES) for x in spans)) if k in ('pii','secret') else 0.0 for k in TASKS})
    result={'examples':len(rows),'teacher':task_metrics(rows,teacher_out),'student':task_metrics(rows,student_out),'rules':task_metrics(rows,rules),'latency_p50_ms':{'teacher_cpu':sorted(teacher_lat)[len(rows)//2],'student_mps':sorted(student_lat)[len(rows)//2]}}
    (ROOT/'reports/teacher_baseline.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['macro_f1'] for k,v in result.items() if k in ('teacher','student','rules')},indent=2))
if __name__=='__main__':main()
