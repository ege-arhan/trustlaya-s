import json
import time
import psutil
import os
from pathlib import Path
from trustlaya.inference import Analyzer
ROOT=Path(__file__).resolve().parents[1]
def bench(name):
    process=psutil.Process(os.getpid());before=process.memory_info().rss
    start=time.perf_counter();a=Analyzer(name);cold=(time.perf_counter()-start)*1000
    rss_delta=(process.memory_info().rss-before)/2**20
    text="Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder."
    for _ in range(3):a.analyze(text)
    times=[]
    for _ in range(20):
        t=time.perf_counter();a.analyze(text);times.append((time.perf_counter()-t)*1000)
    return {"cold_start_ms":cold,"warm_p50_ms":sorted(times)[10],"warm_mean_ms":sum(times)/len(times),"batch":1,"rss_delta_mib_approx":rss_delta}
def main():
    import torch
    result={"pytorch_cpu":bench("torch_cpu"),"onnx_cpu":bench("onnx"),"onnx_int8_cpu":bench("onnx_int8")}
    if torch.backends.mps.is_available(): result["pytorch_mps"]=bench("torch")
    (ROOT/"benchmarks/edge.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
