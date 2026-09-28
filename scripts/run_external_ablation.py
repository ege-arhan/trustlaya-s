"""Post-baseline read-only component diagnostics on the same frozen predictions."""
import json

from trustlaya.evidence import PII_TYPES
from run_external_real import LOCAL, REPORT, metrics, save_json

def main():
    out={}
    for name,file in (("TAB","tab_predictions.json"),("JailbreakLLMs","jailbreak_predictions.json")):
        rows=json.loads((LOCAL/file).read_text())
        for r in rows:
            r["evidence_score"]=max(r["calibrated_score"],.95) if name=="TAB" and any(e["type"] in PII_TYPES for e in r["evidence"]) else r["calibrated_score"]
            r["policy_non_allow"]=float(r["policy_action"]!="ALLOW")
        out[name]={"model_raw":metrics(rows),"model_existing_calibration":metrics(rows,"calibrated_score"),
                   "model_calibration_evidence":metrics(rows,"evidence_score"),
                   "policy_non_allow_operational_proxy":metrics(rows,"policy_non_allow"),
                   "abstain_count":sum(r["uncertain"] for r in rows),
                   "policy_action_counts":{a:sum(r["policy_action"]==a for r in rows) for a in ("ALLOW","REDACT","REVIEW","BLOCK")}}
    save_json(REPORT/"external_ablation.json",out)

if __name__=="__main__":main()
