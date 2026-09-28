"""Fit calibration and freeze one operating point on disjoint DEV halves."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from run_external_real import metrics
from run_v3_external_eval import apply_calibration

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"benchmarks/external/predictions"
OUT=ROOT/"models/trustlaya-s-v4-research"
STRATEGY="WINDOW_MAX"  # chosen on external DEV comparison, before final JLL evaluation


def main():
    rows=json.loads((LOCAL/"v4_window_dev_predictions_v4.json").read_text())
    cal=[];select=[]
    for row in rows:
        bucket=int(hashlib.sha256(row["sha256"].encode()).hexdigest()[:8],16)%2
        (cal if bucket==0 else select).append(row)
    y=np.array([r["gold"] for r in cal],dtype=int)
    raw=np.clip(np.array([r["scores"][STRATEGY]["raw"] for r in cal]),1e-6,1-1e-6)
    assert set(y)=={0,1} and {r["gold"] for r in select}=={0,1}
    fit=LogisticRegression(C=1.0,random_state=20260925,max_iter=1000)
    fit.fit(np.log(raw/(1-raw)).reshape(-1,1),y)
    calibration={"slope":float(fit.coef_[0,0]),"intercept":float(fit.intercept_[0])}
    def prepared(group,adjust):
        return [{"gold":r["gold"],"raw_score":float(apply_calibration([r["scores"][STRATEGY]["raw"]],calibration)[0]
                                                if adjust else r["scores"][STRATEGY]["raw"])} for r in group]
    curve=[]
    for threshold in np.linspace(.01,.99,99):
        m=metrics(prepared(select,True),threshold=float(threshold))
        curve.append({"threshold":round(float(threshold),2),**m})
    feasible=[r for r in curve if r["fpr"] is not None and r["fpr"]<=.25]
    target=[r for r in feasible if r["recall"] is not None and r["recall"]>=.8]
    pool=target or feasible
    chosen=max(pool,key=lambda r:(r["f1"] or 0,r["recall"] or 0,-r["fpr"]))
    protocol={"version":"v4-linear-head-1","strategy":STRATEGY,"calibration":calibration,
              "threshold":chosen["threshold"],"selection_rule":"maximize DEV select F1 subject to FPR<=0.25; prefer recall>=0.80 if feasible",
              "target_recall_feasible":bool(target),"cal_n":len(cal),"select_n":len(select),
              "cal_raw":metrics(prepared(cal,False)),"cal_fitted_in_sample":metrics(prepared(cal,True)),
              "select_raw":metrics(prepared(select,False)),"select_calibrated":metrics(prepared(select,True)),
              "select_operating_point":chosen,"select_curve":curve}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"operating_point.json").write_text(json.dumps({k:protocol[k] for k in ("version","strategy","calibration","threshold","selection_rule","target_recall_feasible","cal_n","select_n")},indent=2)+"\n")
    (ROOT/"reports/v4_dev_selection.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print(json.dumps({k:protocol[k] for k in ("strategy","threshold","target_recall_feasible","cal_n","select_n","select_operating_point")},indent=2),flush=True)


if __name__=="__main__":main()
