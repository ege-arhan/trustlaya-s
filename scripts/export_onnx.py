import json
import argparse
from pathlib import Path
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic,QuantType
from trustlaya.model import TrustLaya,BACKBONE
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",type=Path,default=ROOT/"models/trustlaya-s-v2")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"models/exported/v2")
    args=parser.parse_args()
    if not (args.model_dir/"model.safetensors").exists():
        parser.error(f"Model not found: {args.model_dir / 'model.safetensors'}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model=TrustLaya(BACKBONE,False);model.load(args.model_dir/"model.safetensors");model.eval()
    ids=torch.ones(1,96,dtype=torch.long);mask=torch.ones_like(ids)
    out=args.output_dir/"trustlaya_s.onnx"
    torch.onnx.export(model,(ids,mask),str(out),input_names=["input_ids","attention_mask"],output_names=["risks","severity","action"],dynamic_axes={"input_ids":{0:"batch"},"attention_mask":{0:"batch"},"risks":{0:"batch"},"severity":{0:"batch"},"action":{0:"batch"}},opset_version=17,dynamo=False)
    onnx.checker.check_model(str(out))
    quant=args.output_dir/"trustlaya_s_int8.onnx"
    quantize_dynamic(str(out),str(quant),weight_type=QuantType.QInt8,per_channel=True)
    from safetensors.torch import save_file
    half=args.output_dir/"trustlaya_s_fp16.safetensors"
    save_file({k:v.detach().half().contiguous() for k,v in model.state_dict().items()},str(half))
    report={"model_dir":str(args.model_dir),"student_fp32_bytes":(args.model_dir/"model.safetensors").stat().st_size,"onnx_fp32_bytes":out.stat().st_size,"onnx_int8_bytes":quant.stat().st_size,"student_fp16_bytes":half.stat().st_size}
    (ROOT/"reports/model_sizes_v2.json").write_text(json.dumps(report,indent=2));print(report)
if __name__=="__main__":main()
