"""Download public TrustLaya-S model artifacts into repository layout."""
import shutil
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
def sha256_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()
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
    parser.add_argument('--advanced',action='store_true',help='Download versioned experimental v2 release')
    parser.add_argument('--agentic-candidate',action='store_true',help='Download separate experimental agentic injection-head candidate')
    args=parser.parse_args()
    if args.advanced and args.agentic_candidate:
        parser.error('Choose one versioned artifact set')
    if args.advanced or args.agentic_candidate:
        manifest_name='models/agentic_candidate_manifest.json' if args.agentic_candidate else 'models/advanced_manifest.json'
        manifest=json.loads((ROOT/manifest_name).read_text())
        base='https://github.com/ege-arhan/trustlaya-s/releases/download/'+manifest['version']+'/'
        for name, expected in manifest['files'].items():
            if args.onnx_only and name in ('model.safetensors','trustlaya_s_int8.onnx'):
                continue
            if args.agentic_candidate:
                folder=ROOT/('models/base' if name=='config.json' else 'models/exported/injection_agentic' if name.endswith('.onnx') else 'models/candidates/injection_agentic')
            else:
                folder=ROOT/('models/exported/v2' if name.endswith('.onnx') else 'models/trustlaya-s-v2')
            folder.mkdir(parents=True,exist_ok=True)
            target=folder/name
            if target.exists() and sha256_file(target)==expected['sha256']:
                continue
            temporary=target.with_suffix(target.suffix+'.partial')
            try:
                digest=hashlib.sha256()
                with urllib.request.urlopen(base+name,timeout=60) as remote, temporary.open('wb') as out:
                    for chunk in iter(lambda:remote.read(4*1024*1024),b''):
                        digest.update(chunk);out.write(chunk)
                if digest.hexdigest()!=expected['sha256']:
                    raise ValueError(f'Checksum mismatch for {name}')
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            print(target)
        return
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
