import json
from pathlib import Path
import numpy as np
# Training-only imports stay inside functions so the ONNX runtime needs only numpy.

def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); total=0.0
    for lo,hi in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        mask=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if mask.any(): total+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(total)
def metrics(y,p):
    from sklearn.metrics import brier_score_loss, log_loss
    p=np.clip(p,1e-7,1-1e-7)
    return {"ece":ece(y,p),"brier":float(brier_score_loss(y,p)),"nll":float(log_loss(y,p,labels=[0,1]))}
def fit_temperature(y,logits):
    from sklearn.metrics import log_loss
    from scipy.optimize import minimize_scalar
    from scipy.special import expit
    y=np.asarray(y); logits=np.asarray(logits)
    if len(np.unique(y))<2: return 1.0
    f=lambda t: log_loss(y,expit(logits/t),labels=[0,1])
    return float(minimize_scalar(f,bounds=(0.3,10),method="bounded").x)
def apply(p,t):
    p=np.clip(p,1e-7,1-1e-7)
    return 1/(1+np.exp(-np.log(p/(1-p))/t))
def save(path,temperatures):
    Path(path).write_text(json.dumps(temperatures,indent=2))
def load(path):
    return json.loads(Path(path).read_text())
