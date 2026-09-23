import json
from pathlib import Path
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic,QuantType
from trustlaya.model import TrustLaya,BACKBONE
ROOT=Path(__file__).resolve().parents[1]
def main():
    model=TrustLaya(BACKBONE,False);model.load(ROOT/"models/student/model.safetensors");model.eval()
    ids=torch.ones(1,96,dtype=torch.long);mask=torch.ones_like(ids)
    out=ROOT/"models/exported/trustlaya.onnx"
    torch.onnx.export(model,(ids,mask),str(out),input_names=["input_ids","attention_mask"],output_names=["risks","severity","action"],dynamic_axes={"input_ids":{0:"batch"},"attention_mask":{0:"batch"},"risks":{0:"batch"},"severity":{0:"batch"},"action":{0:"batch"}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(out))
    quant=ROOT/"models/exported/trustlaya_int8.onnx"
    quantize_dynamic(str(out),str(quant),weight_type=QuantType.QInt8,per_channel=True)
    from safetensors.torch import save_file
    half=ROOT/"models/exported/trustlaya_fp16.safetensors"
    save_file({k:v.detach().half().contiguous() for k,v in model.state_dict().items()},str(half))
    report={"student_fp32_bytes":(ROOT/"models/student/model.safetensors").stat().st_size,"onnx_fp32_bytes":out.stat().st_size,"onnx_int8_bytes":quant.stat().st_size,"student_fp16_bytes":half.stat().st_size}
    (ROOT/"reports/model_sizes.json").write_text(json.dumps(report,indent=2));print(report)
if __name__=="__main__":main()
