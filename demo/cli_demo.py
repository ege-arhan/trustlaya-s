import argparse
import json
from trustlaya.inference import Analyzer
p=argparse.ArgumentParser();p.add_argument("--text",default="Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder.");p.add_argument("--metadata",default="{}");p.add_argument("--backend",choices=["torch","onnx"],default="torch")
a=p.parse_args(); result=Analyzer(a.backend).analyze(a.text,json.loads(a.metadata));print(json.dumps(result,ensure_ascii=False,indent=2))
