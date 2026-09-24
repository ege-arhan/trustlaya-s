"""Download public TrustLaya-S model artifacts into repository layout."""
import shutil
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
FILES={
    'model.safetensors':'models/student/model.safetensors',
    'vocab.txt':'models/student/vocab.txt',
    'tokenizer_config.json':'models/student/tokenizer_config.json',
    'special_tokens_map.json':'models/student/special_tokens_map.json',
    'calibration.json':'models/student/calibration.json',
    'trustlaya.onnx':'models/exported/trustlaya.onnx',
    'trustlaya_int8.onnx':'models/exported/trustlaya_int8.onnx',
}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--onnx-only',action='store_true',help='Skip PyTorch weights and INT8 for CI smoke tests')
    args=parser.parse_args()
    for remote,local in FILES.items():
        if args.onnx_only and remote in ('model.safetensors','trustlaya_int8.onnx'):
            continue
        dst=ROOT/local;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():continue
        src=hf_hub_download('ege-arhan/TrustLaya-S',filename=remote)
        shutil.copy2(src,dst);print(dst)
    base=ROOT/'models/base/config.json';base.parent.mkdir(parents=True,exist_ok=True)
    if not base.exists():shutil.copy2(hf_hub_download('ytu-ce-cosmos/turkish-medium-bert-uncased',filename='config.json'),base)
    for filename in ('vocab.txt','tokenizer_config.json','special_tokens_map.json'):
        target=base.parent/filename
        if not target.exists():shutil.copy2(ROOT/'models/student'/filename,target)
if __name__=='__main__':main()
