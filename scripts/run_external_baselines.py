"""Compatible baselines evaluated on exact retained TrustLaya external rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from presidio_analyzer import AnalyzerEngine
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from run_external_real import ROOT, LOCAL, REPORT, RAW, tab_samples, jailbreak_samples, metrics, save_json

JAIL_MODEL = "jackhhao/jailbreak-classifier"
JAIL_REV = "771aa6f1391933e7cba0b21f0f17750c7a74a901"


def by_ids(samples, predictions):
    lookup={s["sample_id"]:s for s in samples}
    return [lookup[r["sample_id"]] for r in predictions]


def main():
    trust_tab=json.loads((LOCAL/"tab_predictions.json").read_text())
    trust_jail=json.loads((LOCAL/"jailbreak_predictions.json").read_text())
    tok=AutoTokenizer.from_pretrained(ROOT/"models/trustlaya-s-v2")
    tab=by_ids(tab_samples(tok)[0],trust_tab)
    jail=by_ids(jailbreak_samples()[0],trust_jail)
    presidio=AnalyzerEngine()
    p_rows=[]
    for i,(s,t) in enumerate(zip(tab,trust_tab)):
        result=presidio.analyze(text=s["text"],language="en",
                                entities=["PERSON","US_SSN","US_PASSPORT","US_DRIVER_LICENSE","CREDIT_CARD","EMAIL_ADDRESS","PHONE_NUMBER","IBAN_CODE","IP_ADDRESS"])
        # Presidio score is recognizer confidence, not a calibrated document risk.
        score=max((x.score for x in result),default=0.0)
        p_rows.append({"sample_id":t["sample_id"],"gold":t["gold"],"score":float(score),
                       "source":t["source"],"text_hash":t["text_hash"]})
        if i and i%500==0:print("Presidio",i,"/",len(tab),flush=True)
    save_json(LOCAL/"presidio_predictions.json",p_rows)
    print("Presidio",metrics(p_rows,"score"),flush=True)

    tokenizer=AutoTokenizer.from_pretrained(JAIL_MODEL,revision=JAIL_REV)
    model=AutoModelForSequenceClassification.from_pretrained(JAIL_MODEL,revision=JAIL_REV).eval()
    j_rows=[]
    for start in range(0,len(jail),16):
        batch=jail[start:start+16]
        tokens=tokenizer([s["text"] for s in batch],padding=True,truncation=True,max_length=512,return_tensors="pt")
        with torch.inference_mode():
            scores=model(**tokens).logits.softmax(-1)[:,1].cpu().tolist()
        for s,t,p in zip(batch,trust_jail[start:start+16],scores):
            j_rows.append({"sample_id":t["sample_id"],"gold":t["gold"],"score":float(p),
                           "source":t["source"],"text_hash":t["text_hash"]})
        if start and start%512==0:print("Jailbreak BERT",start,"/",len(jail),flush=True)
    save_json(LOCAL/"jailbreak_bert_predictions.json",j_rows)
    print("Jailbreak BERT",metrics(j_rows,"score"),flush=True)
    save_json(REPORT/"external_baseline_results.json",{
        "presidio":{"model":"Presidio Analyzer 2.2.364 + spaCy en_core_web_lg 3.8.0", "TAB":metrics(p_rows,"score")},
        "jailbreak_bert":{"model":JAIL_MODEL,"revision":JAIL_REV,"JailbreakLLMs":metrics(j_rows,"score")}})

if __name__=="__main__":main()
