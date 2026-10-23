"""Exact and char-gram near-duplicate audit across V4 research roles."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

from run_external_real import jailbreak_samples, norm

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"benchmarks/external/predictions"


def audit(left,right):
    ln=[norm(r["text"]) for r in left];rn=[norm(r["text"]) for r in right]
    exact=len(set(ln)&set(rn))
    vectorizer=HashingVectorizer(analyzer="char",ngram_range=(4,5),n_features=2**18,
                                 alternate_sign=False,norm="l2")
    x=vectorizer.transform(ln);y=vectorizer.transform(rn)
    distance,_=NearestNeighbors(n_neighbors=1,metric="cosine",algorithm="brute",n_jobs=-1).fit(y).kneighbors(x)
    near=[{"id":left[i].get("id",left[i].get("sample_id")),"max_similarity":round(1-float(d),4)}
          for i,d in enumerate(distance[:,0]) if 1-float(d)>=.85]
    return {"left_n":len(left),"right_n":len(right),"exact_unique":exact,
            "near_left_rows_cosine_gte_0.85":len(near),"near_left_ids":near}


def main():
    train=json.loads((LOCAL/"v4_train_text_private.json").read_text())
    dev=json.loads((LOCAL/"v4_dev_text_private.json").read_text())
    clean={r["sample_id"] for r in json.loads((LOCAL/"v4_frozen_inputs/v3_jailbreak_predictions.json").read_text())}
    test=[r for r in jailbreak_samples()[0] if r["sample_id"] in clean]
    result={"method":"normalized exact; HashingVectorizer char 4-5 gram cosine >=0.85",
            "train_dev":audit(train,dev),"dev_test":audit(dev,test),"train_test":audit(train,test)}
    (ROOT/"reports/v4_data_leakage.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
