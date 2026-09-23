import numpy as np
from trustlaya.calibration import fit_temperature,apply,metrics

def test_temperature():
    y=np.array([0,0,1,1]);z=np.array([-4,-2,2,4]);t=fit_temperature(y,z)
    p=apply(1/(1+np.exp(-z)),t)
    assert 0<t<=10 and 0<=metrics(y,p)['ece']<=1
