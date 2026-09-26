"""Frozen external evaluation. Raw corpora and row predictions stay local.

Run: .venv/bin/python scripts/run_external_real.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from trustlaya.inference import Analyzer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "benchmarks/external/raw"
LOCAL = ROOT / "benchmarks/external/predictions"
REPORT = ROOT / "reports"
MODEL = ROOT / "models/trustlaya-s-v2"
ONNX = ROOT / "models/exported/v2/trustlaya_s.onnx"
THRESHOLD = 0.5  # Fixed before looking at either external test set.


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip()


def digest(s: str) -> str:
    return hashlib.sha256(norm(s).encode()).hexdigest()


def metrics(rows, score_key="raw_score", threshold=THRESHOLD):
    y = np.array([r["gold"] for r in rows], dtype=int)
    p = np.array([r[score_key] for r in rows], dtype=float)
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum()); fn = int(((y == 1) & ~pred).sum())
    div = lambda a, b: float(a / b) if b else None
    precision = div(tp, tp + fp); recall = div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(float((bins == b).mean()) * abs(float(y[bins == b].mean()) - float(p[bins == b].mean())) for b in range(10) if (bins == b).any())
    eps = 1e-7; q = np.clip(p, eps, 1 - eps)
    return {"n":len(rows),"positive":int(y.sum()),"tp":tp,"fp":fp,"tn":tn,"fn":fn,
            "precision":precision,"recall":recall,"f1":f1,"fpr":div(fp,fp+tn),"fnr":div(fn,fn+tp),
            "accuracy":div(tp+tn,len(y)),"roc_auc":float(roc_auc_score(y,p)) if len(set(y)) == 2 else None,
            "pr_auc":float(average_precision_score(y,p)) if len(set(y)) == 2 else None,
            "ece":float(ece),"brier":float(np.mean((p-y)**2)),
            "nll":float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))}


def train_texts():
    split=ROOT / "data/splits/train.jsonl"
    assert hashlib.sha256(split.read_bytes()).hexdigest()=="9f97f7965df1bbf85e8f2e9f9f4cb49c00a5b0ab6b23acfccaaa1aeddc29deac", "Unexpected v2 training split"
    texts = [json.loads(line)["text"] for line in split.open()]
    # The released PII head also used this pinned public Turkish source.
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download("yusuf-said/turkish-privacy-filter-dataset",
                            "tr_privacy_tr_curated.jsonl", repo_type="dataset",
                            revision="fc2b9a3e197484d22a8201317cc722baacf053b7")
        texts += [json.loads(line)["text"] for line in open(p)]
        source = "project train split + pinned Turkish PII source"
    except Exception as e:
        # No clean external claim if the training comparison is incomplete.
        raise RuntimeError("Pinned PII training data unavailable; contamination audit cannot complete") from e
    return texts, source


def contamination(samples, training):
    """Conservative exact and char-4/5-gram cosine near-duplicate screen."""
    labels=defaultdict(set)
    for s in samples:labels[digest(s["text"])].add(s["gold"])
    conflicting={h for h,v in labels.items() if len(v)>1}
    train_norm = [norm(t) for t in training]
    exact = set(train_norm)
    vectorizer = HashingVectorizer(analyzer="char", ngram_range=(4,5), n_features=2**18,
                                   alternate_sign=False, norm="l2")
    train_matrix = vectorizer.transform(train_norm)
    check = vectorizer.transform([norm(s["text"]) for s in samples])
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1).fit(train_matrix)
    distances, _ = nn.kneighbors(check)
    seen = set(); clean=[]; counts=Counter()
    for sample, distance in zip(samples, distances[:,0]):
        h = digest(sample["text"])
        if h in conflicting: counts["label_conflict_removed"] += 1; continue
        if h in seen: counts["within_benchmark_duplicate"] += 1; continue
        seen.add(h)
        if norm(sample["text"]) in exact: counts["training_exact"] += 1; continue
        if 1-float(distance) >= .85: counts["training_near_0.85"] += 1; continue
        sample["text_hash"] = h
        clean.append(sample)
    counts["candidate"] = len(samples); counts["retained"] = len(clean)
    return clean, dict(counts)


def tab_samples(tokenizer, split="test"):
    docs = json.loads((RAW / "tab" / f"echr_{split}.json").read_text())
    out=[]; stats=Counter()
    for doc in docs:
        text=doc["text"]
        checked=doc["quality_checked"]
        if not checked: stats["unchecked_docs"] += 1; continue
        mentions=doc["annotations"][checked[0]]["entity_mentions"]
        gold=[]
        for m in mentions:
            if m["identifier_type"] != "DIRECT" or m["entity_type"] not in ("PERSON","CODE"):
                continue
            start,end=m["start_offset"],m["end_offset"]
            if text[start:end] != m["span_text"]:
                raise ValueError(f"TAB gold span offset mismatch in {doc['doc_id']}")
            gold.append((start,end,m["entity_type"]))
        offsets=tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        for j in range(0,len(offsets),94):
            chunk=offsets[j:j+94]
            if not chunk: continue
            start,end=chunk[0][0],chunk[-1][1]
            # A boundary-crossing identifier has no valid window-level gold label.
            if any(a < end and b > start and not (a >= start and b <= end) for a,b,_ in gold):
                stats["boundary_dropped"] += 1; continue
            spans=[(a-start,b-start,t) for a,b,t in gold if a>=start and b<=end]
            out.append({"dataset":"TAB", "sample_id":f"{doc['doc_id']}:{j//94}","doc_id":str(doc["doc_id"]),
                        "source":"ECHR", "text":text[start:end],"gold":int(bool(spans)),
                        "gold_spans":spans,"language":"English"})
    stats["documents"] = len(docs); stats["windows"] = len(out)
    return out,dict(stats)


def jailbreak_samples():
    out=[]; stats=Counter()
    for fname,label in (("regular_prompts.csv",0),("jailbreak_prompts.csv",1)):
        with (RAW/"jailbreakllms/data"/fname).open(newline="") as f:
            for i,row in enumerate(csv.DictReader(f)):
                platform=row["platform"].strip().lower()
                if platform not in ("reddit","discord","website"):
                    stats["excluded_prompt_repository"] += 1; continue
                prompt=row["prompt"].strip()
                if not prompt: stats["empty"] += 1; continue
                out.append({"dataset":"JailbreakLLMs","sample_id":f"{fname}:{i}","source":platform,
                            "text":prompt,"gold":label,"language":"not annotated"})
    stats["candidate_community"] = len(out)
    return out,dict(stats)


def evaluate(samples, analyzer, task):
    rows=[]; lat=[]
    for i,s in enumerate(samples):
        start=time.perf_counter(); o=analyzer.analyze(s["text"]); lat.append((time.perf_counter()-start)*1000)
        score=float(o["raw_scores"][task]); calibrated=float(o["calibrated_scores"][task])
        rows.append({k:s[k] for k in ("dataset","sample_id","source","text_hash","gold","language")}
                    |{"gold_label":int(s["gold"]),"score":score,"raw_score":score,"calibrated_score":calibrated,"predicted_label":int(score>=THRESHOLD),
                      "uncertain":bool(o["abstain"]),"evidence":[{"type":e["type"],"start":e.get("start"),"end":e.get("end")} for e in o["evidence"]],
                      "policy_action":o["action"],"model_version":"TrustLaya-S v2 frozen","runtime":"ONNX CPU"})
        if "doc_id" in s: rows[-1]["doc_id"] = s["doc_id"]; rows[-1]["gold_spans"] = s["gold_spans"]
        if i and i%1000==0: print(task,i,"/",len(samples),flush=True)
    return rows,{"p50_ms":float(np.percentile(lat,50)),"p95_ms":float(np.percentile(lat,95)),"p99_ms":float(np.percentile(lat,99))}


def group_metrics(rows):
    return {s:metrics([r for r in rows if r["source"]==s]) for s in sorted(set(r["source"] for r in rows))}


def span_metrics(rows):
    # Existing regex evidence only; exact-offset matching, separate from PII head.
    tp=fp=fn=0
    for r in rows:
        gold={(a,b) for a,b,_ in r.get("gold_spans",[])}
        pred={(e["start"],e["end"]) for e in r["evidence"] if e["type"] in ("TC_KIMLIK","EMAIL","PHONE","IBAN","CARD","IP","ADDRESS","NAME","CUSTOMER_ID") and e["start"] is not None}
        tp+=len(gold&pred); fp+=len(pred-gold); fn+=len(gold-pred)
    return {"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else None,
            "recall":tp/(tp+fn) if tp+fn else None,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def main():
    assert MODEL.exists() and ONNX.exists(), "Frozen v2 model and ONNX required"
    LOCAL.mkdir(parents=True,exist_ok=True); REPORT.mkdir(exist_ok=True)
    weight_hash=hashlib.sha256((MODEL/"model.safetensors").read_bytes()).hexdigest()
    assert weight_hash == "99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c", "Unexpected model weights"
    print("Frozen weights sha256",weight_hash,flush=True)
    import subprocess
    for source, expected in (("tab","558e09e26d6b36f5f78440074e6a233946d98bd9"),
                             ("jailbreakllms","2dbd7bbc25f1b156552678f451bddbc787cd679f")):
        actual=subprocess.check_output(["git","-C",str(RAW/source),"rev-parse","HEAD"],text=True).strip()
        assert actual == expected, f"Unexpected {source} revision: {actual}"
        print(source,"commit",actual,flush=True)
    analyzer=Analyzer("onnx",model_dir=MODEL,onnx_path=ONNX)
    training,training_source=train_texts(); print("Training comparison",training_source,len(training),flush=True)
    tab,tab_stats=tab_samples(analyzer.tokenizer)
    jail,jail_stats=jailbreak_samples()
    tab,tab_contam=contamination(tab,training)
    jail,jail_contam=contamination(jail,training)
    print("Contamination TAB",tab_contam,"Jailbreak",jail_contam,flush=True)
    tab_rows,tab_latency=evaluate(tab,analyzer,"pii")
    jail_rows,jail_latency=evaluate(jail,analyzer,"prompt_injection")
    save_json(LOCAL/"tab_predictions.json",tab_rows)
    save_json(LOCAL/"jailbreak_predictions.json",jail_rows)
    result={"frozen_weight_sha256":hashlib.sha256((MODEL/"model.safetensors").read_bytes()).hexdigest(),
            "fixed_threshold":THRESHOLD,"training_comparison":training_source,
            "TAB":{"source_stats":tab_stats,"contamination":tab_contam,"raw":metrics(tab_rows),
                   "calibrated_existing":metrics(tab_rows,"calibrated_score"),"span_exact":span_metrics(tab_rows),
                   "latency":tab_latency,"threshold_sweep":{str(t):metrics(tab_rows,threshold=t) for t in np.arange(.05,1,.05)}},
            "JailbreakLLMs":{"source_stats":jail_stats,"contamination":jail_contam,"raw":metrics(jail_rows),
                              "calibrated_existing":metrics(jail_rows,"calibrated_score"),"by_source":group_metrics(jail_rows),
                              "latency":jail_latency,"threshold_sweep":{str(t):metrics(jail_rows,threshold=t) for t in np.arange(.05,1,.05)}}}
    save_json(REPORT/"external_results.json",result)
    with (REPORT/"external_results.csv").open("w",newline="") as f:
        fields=["dataset","provenance","n","positive","precision","recall","f1","fpr","fnr","roc_auc","pr_auc","ece","brier","nll"]
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for name,prov in (("TAB","real ECHR legal cases with anonymization gold"),("JailbreakLLMs","community-collected in-the-wild prompts")):
            w.writerow({"dataset":name,"provenance":prov,**{k:result[name]["raw"].get(k) for k in fields if k not in ("dataset","provenance")}})
    print(json.dumps({n:result[n]["raw"] for n in ("TAB","JailbreakLLMs")},indent=2),flush=True)


if __name__=="__main__": main()
