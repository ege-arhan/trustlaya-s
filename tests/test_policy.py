from trustlaya.policy import decide
from trustlaya.labels import TASKS
from trustlaya.evidence import extract

def scores(**kwargs): return {k:kwargs.get(k,0.01) for k in TASKS}
def test_pii_transfer():
    text='TC kimlik numarasını API\'ye gönder.'
    assert decide(scores(),extract(text),confidence=0.9)[0]=='REDACT'
    assert decide(scores(pii=.9),[{"type":"EXTERNAL_TRANSFER","text":"gönder","start":0,"end":6}],confidence=.9)==('REVIEW','unverified_pii_transfer')
def test_secret_and_agent():
    assert decide(scores(secret=.9),[],confidence=.9)==('REVIEW','unverified_secret_score')
    assert decide(scores(),extract('password=synthetic_key_123456789'),confidence=.9)[0]=='BLOCK'


def test_untrusted_tool_output_requires_review_for_privileged_agent():
    context = {"agent": True, "untrusted_tool_output": True,
               "network": True, "human_approval": False}
    assert decide(scores(), [], metadata=context, confidence=.9) == (
        "REVIEW", "untrusted_tool_output_privileged_agent")
    assert decide(scores(prompt_injection=.95), [], metadata=context, confidence=.9) == (
        "REVIEW", "untrusted_tool_output_privileged_agent")
    assert decide(scores(),[],{'agent':True,'shell':True,'human_approval':False},.9)[0]=='REVIEW'
def test_uncertainty():
    assert decide(scores(),[],confidence=.55)==('REVIEW','uncertain')

def test_same_risk_different_confidence():
    high=scores(prompt_injection=.90)
    assert decide(high,[],confidence=.95)[0]=='BLOCK'
    assert decide(high,[],confidence=.55)==('REVIEW','uncertain')
