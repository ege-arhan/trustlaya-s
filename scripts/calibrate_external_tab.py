"""Optional post-baseline TAB-dev temperature experiment; never edits deployed model."""
import json

import numpy as np
from scipy.special import logit

from trustlaya.calibration import apply, fit_temperature
from trustlaya.inference import Analyzer
from run_external_real import ROOT, RAW, LOCAL, REPORT, tab_samples, train_texts, contamination, evaluate, metrics, save_json


def main():
    if not (REPORT/"external_results.json").exists():
        raise RuntimeError("Frozen external baseline must be saved first")
    analyzer=Analyzer("onnx",model_dir=ROOT/"models/trustlaya-s-v2",
                      onnx_path=ROOT/"models/exported/v2/trustlaya_s.onnx")
    dev,source_stats=tab_samples(analyzer.tokenizer,"dev")
    training,_=train_texts()
    dev,contam=contamination(dev,training)
    # Exclude exact dev/test overlap without consulting test labels.
    test_hashes={r["text_hash"] for r in json.loads((LOCAL/"tab_predictions.json").read_text())}
    dev=[s for s in dev if s["text_hash"] not in test_hashes]
    rows,_=evaluate(dev,analyzer,"pii")
    y=np.array([r["gold"] for r in rows]);p=np.array([r["raw_score"] for r in rows])
    temp=fit_temperature(y,logit(np.clip(p,1e-7,1-1e-7)))
    for r in rows:r["dev_fitted_score"]=float(apply(r["raw_score"],temp))
    test=json.loads((LOCAL/"tab_predictions.json").read_text())
    for r in test:r["dev_fitted_score"]=float(apply(r["raw_score"],temp))
    result={"temperature":temp,"dev_source_stats":source_stats,"dev_contamination":contam,
            "dev_n_after_test_overlap":len(rows),
            "dev_raw":metrics(rows),"dev_fitted":metrics(rows,"dev_fitted_score"),
            "test_raw":metrics(test),"test_dev_fitted":metrics(test,"dev_fitted_score"),
            "note":"Post-baseline exploratory calibration; no model or deployed temperature changed."}
    save_json(REPORT/"external_tab_dev_calibration.json",result)
    print(json.dumps({"temperature":temp,"dev_n":len(rows),"test_dev_fitted":result["test_dev_fitted"]},indent=2))

if __name__=="__main__":main()
