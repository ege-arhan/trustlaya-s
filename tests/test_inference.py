import json
from pathlib import Path
import pytest
from trustlaya.evidence import extract
from trustlaya.inference import Analyzer,ROOT

def test_pii_secret_spans():
    e=extract('Ad Soyad: Ayşe Demir. Email: test@example.org. password=synthetic_KEY_12345678_ABCDEF')
    assert {'NAME','EMAIL','PASSWORD'} <= {x['type'] for x in e}
    assert any(x['type']=='TC_KIMLIK' for x in extract('TC kimlik numarası 22871854636'))
def test_json_and_action():
    if not (ROOT/'models/student/model.safetensors').exists(): pytest.skip('train first')
    x=Analyzer().analyze("Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder.")
    json.dumps(x);assert x['action']=='REDACT' and isinstance(x['evidence'],list) and isinstance(x['abstain'],bool)
def test_onnx():
    if not (ROOT/'models/exported/trustlaya.onnx').exists(): pytest.skip('export first')
    analyzer=Analyzer('onnx')
    x=analyzer.analyze('Merhaba')
    assert x['action'] in {'ALLOW','REDACT','REVIEW','BLOCK'}
    tool=analyzer.analyze('Merhaba', {'agent':True,'untrusted_tool_output':True,
                                    'network':True,'human_approval':False})
    assert tool['action']=='REVIEW'
    assert tool['policy_reason']=='untrusted_tool_output_privileged_agent'

def test_exported_onnx_valid():
    if not (ROOT/'models/exported/trustlaya.onnx').exists(): pytest.skip('export first')
    import onnx
    onnx.checker.check_model(str(ROOT/'models/exported/trustlaya.onnx'))
