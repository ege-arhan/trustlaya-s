import torch
from trustlaya.model import TrustLaya,BACKBONE
from transformers import AutoTokenizer

def test_creation_and_tokenizer():
    t=AutoTokenizer.from_pretrained(BACKBONE);m=TrustLaya(BACKBONE,False)
    x=t('TC kimlik numarası',return_tensors='pt')
    with torch.inference_mode(): r,s,a=m(x['input_ids'],x['attention_mask'])
    assert r.shape==(1,9) and s.shape==(1,4) and a.shape==(1,4)
