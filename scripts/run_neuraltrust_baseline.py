"""Independent-model-card jailbreak baseline on the exact external rows.

Its private training data prevents a verified contamination exclusion; this is
reported explicitly and the comparison is not treated as a clean leaderboard.
"""
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

from run_external_real import LOCAL, REPORT, jailbreak_samples, metrics, save_json

MODEL="NeuralTrust/prompt-guard-oss-small"
REV="40e5c56b68a1b081484c3e56c9bee726e2749138"

def main():
    trust=json.loads((LOCAL/"jailbreak_predictions.json").read_text())
    original={s["sample_id"]:s for s in jailbreak_samples()[0]}
    # This revision declares the newer TokenizersBackend class, unavailable in
    # the repository's pinned Transformers 4.x. The standard fast tokenizer can
    # load the same tokenizer.json without changing the model or vocabulary.
    tokenizer=PreTrainedTokenizerFast(tokenizer_file=hf_hub_download(MODEL,"tokenizer.json",revision=REV),
                                      bos_token="<bos>",eos_token="<eos>",pad_token="<pad>",unk_token="<unk>",
                                      cls_token="<bos>",sep_token="<eos>",mask_token="<mask>")
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REV).eval()
    rows=[]
    for start in range(0,len(trust),16):
        chunk=trust[start:start+16]
        tokens=tokenizer([original[r["sample_id"]]["text"] for r in chunk],padding=True,truncation=True,max_length=512,return_tensors="pt")
        with torch.inference_mode():scores=model(**tokens).logits.softmax(-1)[:,1].cpu().tolist()
        rows += [{"sample_id":r["sample_id"],"text_hash":r["text_hash"],"source":r["source"],
                  "gold":r["gold"],"score":float(s)} for r,s in zip(chunk,scores)]
        if start and start%512==0:print("NeuralTrust",start,"/",len(trust),flush=True)
    save_json(LOCAL/"neuraltrust_predictions.json",rows)
    result={"model":MODEL,"revision":REV,"license":"MIT","metrics":metrics(rows,"score"),
            "training_contamination_status":"UNKNOWN: model card states private fine-tuning dataset; cannot verify overlap with this benchmark"}
    save_json(REPORT/"external_neuraltrust_baseline.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
