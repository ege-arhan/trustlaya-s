"""Audit BERT baseline's published training corpus against evaluated prompts."""
import csv
import json

from huggingface_hub import hf_hub_download

from run_external_real import LOCAL, REPORT, jailbreak_samples, contamination, metrics, save_json

DATASET="jackhhao/jailbreak-classification"
REV="2f2ceeb39658696fd3f462403562b6eea5306287"

def main():
    train=[]
    for file in ("balanced/jailbreak_dataset_train_balanced.csv","default/jailbreak_dataset_train.csv"):
        path=hf_hub_download(DATASET,file,repo_type="dataset",revision=REV)
        train.extend(row["prompt"] for row in csv.DictReader(open(path,newline="")))
    original={s["sample_id"]:s for s in jailbreak_samples()[0]}
    trust=json.loads((LOCAL/"jailbreak_predictions.json").read_text())
    bert=json.loads((LOCAL/"jailbreak_bert_predictions.json").read_text())
    samples=[original[r["sample_id"]] for r in trust]
    clean,stats=contamination(samples,train)
    ids={s["sample_id"] for s in clean}
    trust_clean=[r for r in trust if r["sample_id"] in ids]
    bert_clean=[r for r in bert if r["sample_id"] in ids]
    assert [r["sample_id"] for r in trust_clean]==[r["sample_id"] for r in bert_clean]
    result={"baseline_training_dataset":DATASET,"revision":REV,"training_rows_loaded":len(train),
            "overlap":stats,"caveat":"Both baseline training and evaluation derive partly from the same TrustAIRLab source; nonoverlap is not source-independent.",
            "same_clean_rows":{"TrustLaya-S":metrics(trust_clean),"BERT":metrics(bert_clean,"score")}}
    save_json(REPORT/"external_baseline_overlap.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
