import argparse
import json
from pathlib import Path
from trustlaya.inference import Analyzer
p=argparse.ArgumentParser();p.add_argument("--text",default="Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder.");p.add_argument("--metadata",default="{}");p.add_argument("--backend",choices=["torch","torch_cpu","onnx","onnx_int8"],default="torch")
p.add_argument("--model-dir",type=Path);p.add_argument("--onnx",type=Path)
a=p.parse_args(); result=Analyzer(a.backend,model_dir=a.model_dir,onnx_path=a.onnx).analyze(a.text,json.loads(a.metadata));print(json.dumps(result,ensure_ascii=False,indent=2))
