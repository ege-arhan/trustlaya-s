import argparse
import json
from pathlib import Path
from trustlaya.dataset import write_splits,read_rows
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--count",type=int,default=10000); a=p.parse_args()
    counts=write_splits(ROOT/"data/splits",a.count)
    splits={name:read_rows(ROOT/f"data/splits/{name}.jsonl") for name in counts}
    families=[{r["family"] for r in rows} for rows in splits.values()]
    assert not any(families[i]&families[j] for i in range(3) for j in range(i+1,3))
    assert sum(counts.values())>=10000 if a.count>=10000 else True
    (ROOT/"data/processed/validation.json").write_text(json.dumps({"counts":counts,"family_overlap":0},indent=2))
    print(counts)
if __name__=="__main__": main()
