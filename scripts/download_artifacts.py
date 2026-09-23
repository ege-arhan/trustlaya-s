"""Download public TrustLaya-S model artifacts into repository layout."""
import shutil
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
    for remote,local in FILES.items():
        dst=ROOT/local;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():continue
        src=hf_hub_download('xzwq/TrustLaya-S',filename=remote)
        shutil.copy2(src,dst);print(dst)
    base=ROOT/'models/base/config.json';base.parent.mkdir(parents=True,exist_ok=True)
    if not base.exists():shutil.copy2(hf_hub_download('ytu-ce-cosmos/turkish-medium-bert-uncased',filename='config.json'),base)
if __name__=='__main__':main()
