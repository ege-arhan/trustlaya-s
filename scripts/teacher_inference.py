"""Optional Laya soft-target collection. Never treats teacher as ground truth."""
import argparse
import json
from pathlib import Path
from trustlaya.dataset import read_rows
from trustlaya.labels import TASKS
ROOT=Path(__file__).resolve().parents[1]
QUESTIONS={k:{"type":"noul","instructions":f"Does this text contain {k.replace('_',' ')}?"} for k in TASKS}
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=256);a=p.parse_args()
    import laya
    agent=laya.load(str(ROOT/'models/teacher'))
    rows=read_rows(ROOT/'data/splits/train.jsonl')[:a.limit]
    out=[]
    for i,row in enumerate(rows):
        result=agent.predict({'text':row['text']},QUESTIONS)
        out.append({'text':row['text'],'teacher':{k:result['answers'][k]['noul'] for k in TASKS}})
        if (i+1)%25==0:print(i+1,flush=True)
    dest=ROOT/'data/generated/teacher_soft.jsonl'
    with dest.open('w') as f:
        for item in out:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print('saved',len(out))
if __name__=='__main__':main()
