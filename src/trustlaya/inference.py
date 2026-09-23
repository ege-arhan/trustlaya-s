import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .model import TrustLaya,BACKBONE
from .labels import TASKS,ACTIONS,SEVERITIES
from .evidence import extract,PII_TYPES,SECRET_TYPES
from .policy import decide
from .calibration import apply
from .uncertainty import should_abstain
from .utils import device,normalize
ROOT=Path(__file__).resolve().parents[2]
class Analyzer:
    def __init__(self,backend="torch",model_dir=None):
        self.model_dir=Path(model_dir or ROOT/"models/student")
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
        self.temperatures=json.loads((self.model_dir/"calibration.json").read_text()) if (self.model_dir/"calibration.json").exists() else {t:1.0 for t in TASKS}
        self.backend=backend
        if backend in ("onnx","onnx_int8","onnx_int8_pc"):
            import onnxruntime as ort
            self.session=ort.InferenceSession(str(ROOT/("models/exported/trustlaya_int8_pc.onnx" if backend=="onnx_int8_pc" else "models/exported/trustlaya_int8.onnx" if backend=="onnx_int8" else "models/exported/trustlaya.onnx")),providers=["CPUExecutionProvider"])
        else:
            self.torch_device=torch.device("cpu") if backend=="torch_cpu" else device()
            self.model=TrustLaya(BACKBONE,pretrained=False)
            self.model.load(self.model_dir/"model.safetensors")
            self.model.to(self.torch_device).eval()
    def analyze(self,text,metadata=None):
        tokens=self.tokenizer(normalize(text),return_tensors="pt",truncation=True,max_length=96,padding="max_length")
        if self.backend in ("onnx","onnx_int8","onnx_int8_pc"):
            output=self.session.run(None,{k:v.numpy() for k,v in tokens.items() if k in ("input_ids","attention_mask")})
            raw=output[0][0]; sev=output[1][0]; act=output[2][0]
            import numpy as np
            prob=1/(1+np.exp(-raw))
        else:
            with torch.inference_mode():
                output=self.model(tokens["input_ids"].to(self.torch_device),tokens["attention_mask"].to(self.torch_device))
            prob=torch.sigmoid(output[0])[0].cpu().numpy(); sev=output[1][0].cpu().numpy(); act=output[2][0].cpu().numpy()
        scores={task:float(apply(prob[i],self.temperatures.get(task,1.0))) for i,task in enumerate(TASKS)}
        evidence=extract(text)
        if any(e["type"] in PII_TYPES for e in evidence): scores["pii"]=max(scores["pii"],0.95)
        if any(e["type"] in SECRET_TYPES for e in evidence): scores["secret"]=max(scores["secret"],0.95)
        # Separate categorical certainty from risk magnitude; this is not correctness calibration.
        import numpy as np
        def top_softmax(x):
            z=np.exp(x-np.max(x));return float(np.max(z/z.sum()))
        conf=min(top_softmax(sev),top_softmax(act))
        action,reason=decide(scores,evidence,metadata,conf)
        out={**scores,"severity":SEVERITIES[int(sev.argmax())],"model_action":ACTIONS[int(act.argmax())],"action":action,"policy_reason":reason,"confidence":round(conf,4),"abstain":reason=="uncertain","evidence":evidence}
        return out
