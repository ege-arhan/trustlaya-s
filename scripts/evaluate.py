import json
from pathlib import Path
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from trustlaya.dataset import read_rows
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS
from trustlaya.evidence import extract,PII_TYPES,SECRET_TYPES
ROOT=Path(__file__).resolve().parents[1]
def measures(y,p):
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    tn=sum(a==0 and b==0 for a,b in zip(y,p));fp=sum(a==0 and b==1 for a,b in zip(y,p));fn=sum(a==1 and b==0 for a,b in zip(y,p));tp=sum(a==1 and b==1 for a,b in zip(y,p))
    return {"accuracy":accuracy_score(y,p),"precision":precision,"recall":recall,"f1":f1,"false_positive_rate":fp/max(1,fp+tn),"false_negative_rate":fn/max(1,fn+tp),"support":sum(y)}
def main():
    rows=read_rows(ROOT/"data/splits/test.jsonl");model=Analyzer();pred=[model.analyze(r["text"]) for r in rows]
    report={"examples":len(rows),"tasks":{},"baseline_rules":{}}
    for key in TASKS:
        y=[r["labels"][key] for r in rows];p=[int(v[key]>=0.5) for v in pred]
        report["tasks"][key]=measures(y,p)
        if key in ("pii","secret"):
            types=PII_TYPES if key=="pii" else SECRET_TYPES
            report["baseline_rules"][key]=measures(y,[int(any(e["type"] in types for e in extract(r["text"]))) for r in rows])
    report["macro_f1"]=sum(v["f1"] for v in report["tasks"].values())/len(TASKS)
    report["accuracy"]=sum(v["accuracy"] for v in report["tasks"].values())/len(TASKS)
    (ROOT/"reports/evaluation.json").write_text(json.dumps(report,indent=2));print(json.dumps({"examples":len(rows),"macro_f1":report["macro_f1"],"accuracy":report["accuracy"]},indent=2))
if __name__=="__main__":main()
