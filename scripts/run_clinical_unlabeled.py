"""Unlabeled clinical flag-rate diagnostic; never reports PHI FPR/F1."""
import hashlib
import json
from pathlib import Path

from trustlaya.inference import Analyzer
from run_external_real import ROOT, RAW, REPORT, save_json

EXPECTED="15e67ea86ed0678425ec0948032f3fe9f77f30db"

def main():
    import subprocess
    repo=RAW/"glaucoma"
    revision=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    assert revision==EXPECTED,revision
    files=sorted((repo/"Deidentified_Notes").glob("*.txt"))
    analyzer=Analyzer("onnx",model_dir=ROOT/"models/trustlaya-s-v2",
                      onnx_path=ROOT/"models/exported/v2/trustlaya_s.onnx")
    docs_flagged=0;windows=0;windows_flagged=0;evidence_windows=0
    for i,path in enumerate(files):
        text=path.read_text(errors="replace")
        offsets=analyzer.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,verbose=False)["offset_mapping"]
        doc_flag=False
        for start in range(0,len(offsets),94):
            chunk=offsets[start:start+94]
            if not chunk:continue
            out=analyzer.analyze(text[chunk[0][0]:chunk[-1][1]])
            flag=out["raw_scores"]["pii"]>=.5
            windows+=1;windows_flagged+=int(flag);doc_flag|=flag
            evidence_windows+=int(bool(out["evidence"]))
        docs_flagged+=int(doc_flag)
        if i and i%100==0:print("clinical",i,"/",len(files),flush=True)
    result={"dataset":"Glaucoma_Med_Dataset Deidentified_Notes","source_revision":revision,
            "documents":len(files),"windows":windows,"pii_raw_flagged_documents":docs_flagged,
            "pii_raw_flagged_windows":windows_flagged,"any_evidence_windows":evidence_windows,
            "gold_phi_available":False,
            "interpretation":"Unlabeled flag rates only. De-identified clinical notes may contain surrogate or residual identifiers; FPR, recall and F1 are undefined."}
    save_json(REPORT/"external_clinical_unlabeled.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
