import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from .model import TrustLaya,BACKBONE
from .labels import TASKS,ACTIONS,SEVERITIES
from .evidence import extract,PII_TYPES,SECRET_TYPES
from .policy import decide, DEFAULT
from .calibration import apply
from .utils import device,normalize
from .agent_risk import analyze_agent
from .risk_fusion import fuse
ROOT=Path(__file__).resolve().parents[2]
class Analyzer:
    def __init__(self,backend="torch",model_dir=None,onnx_path=None):
        self.model_dir=Path(model_dir or ROOT/"models/student")
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
        self.temperatures=json.loads((self.model_dir/"calibration.json").read_text()) if (self.model_dir/"calibration.json").exists() else {t:1.0 for t in TASKS}
        import yaml
        policy_file=self.model_dir/"policy.yaml"
        self.policy=yaml.safe_load((policy_file if policy_file.exists() else DEFAULT).read_text())
        self.backend=backend
        if backend in ("onnx","onnx_int8","onnx_int8_pc"):
            import onnxruntime as ort
            path=onnx_path or ROOT/("models/exported/trustlaya_int8_pc.onnx" if backend=="onnx_int8_pc" else "models/exported/trustlaya_int8.onnx" if backend=="onnx_int8" else "models/exported/trustlaya.onnx")
            self.session=ort.InferenceSession(str(path),providers=["CPUExecutionProvider"])
        else:
            self.torch_device=torch.device("cpu") if backend=="torch_cpu" else device()
            self.model=TrustLaya(BACKBONE,pretrained=False)
            self.model.load(self.model_dir/"model.safetensors")
            self.model.to(self.torch_device).eval()
    def analyze(self,text,metadata=None,session=None,policy_override=None):
        if not isinstance(text, str): raise TypeError("text must be a string")
        if metadata is not None and not isinstance(metadata, dict): raise TypeError("metadata must be a mapping")
        if policy_override is not None and not isinstance(policy_override,dict): raise TypeError("policy_override must be a mapping")
        metadata=metadata or {}
        policy=self.policy.copy()
        for key,value in (policy_override or {}).items():
            if key not in policy or isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=1:
                raise ValueError(f"Invalid policy threshold: {key}")
            policy[key]=float(value)
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
        raw_scores={task:float(prob[i]) for i,task in enumerate(TASKS)}
        calibrated_scores={task:float(apply(prob[i],self.temperatures.get(task,1.0))) for i,task in enumerate(TASKS)}
        scores=calibrated_scores.copy()
        evidence=extract(text)
        if any(e["type"] in PII_TYPES for e in evidence): scores["pii"]=max(scores["pii"],0.95)
        if any(e["type"] in SECRET_TYPES for e in evidence): scores["secret"]=max(scores["secret"],0.95)
        # Separate categorical certainty from risk magnitude; this is not correctness calibration.
        import numpy as np
        def top_softmax(x):
            z=np.exp(x-np.max(x));return float(np.max(z/z.sum()))
        conf=min(top_softmax(sev),top_softmax(act))
        agent_risk=analyze_agent(metadata)
        fusion=fuse(scores,evidence,metadata,agent_risk)
        session_assessment=session.preview(text,scores,evidence,metadata) if session is not None else None
        action,reason=decide(scores,evidence,metadata,conf,config=policy,agent_risk=agent_risk,fusion=fusion,session=session_assessment)
        if session is not None: session.record(session_assessment,action)
        public_session={k:v for k,v in session_assessment.items() if k!="_summary"} if session_assessment else None
        out={**scores,"raw_scores":raw_scores,"calibrated_scores":calibrated_scores,
             "risk_fusion":fusion,"agent_risk":agent_risk,"session_risk":public_session,
             "severity":SEVERITIES[int(sev.argmax())],"model_action":ACTIONS[int(act.argmax())],
             "action":action,"policy_reason":reason,"confidence":round(conf,4),
             "abstain":reason=="uncertain","evidence":evidence}
        return out
