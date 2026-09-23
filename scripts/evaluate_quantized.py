import json
from pathlib import Path
from sklearn.metrics import f1_score
from trustlaya.dataset import read_rows
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=read_rows(ROOT/'data/splits/test.jsonl')[:256]
    analyzers={key:Analyzer(key) for key in ('onnx','onnx_int8')}
    outputs={key:[analyzer.analyze(r['text']) for r in rows] for key,analyzer in analyzers.items()}
    result={'examples':len(rows),'tasks':{},'action_disagreement':sum(a['action']!=b['action'] for a,b in zip(outputs['onnx'],outputs['onnx_int8']))/len(rows)}
    for task in TASKS:
        y=[r['labels'][task] for r in rows]
        result['tasks'][task]={key:f1_score(y,[int(p[task]>=.5) for p in preds],zero_division=0) for key,preds in outputs.items()}
    (ROOT/'reports/quantization.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
