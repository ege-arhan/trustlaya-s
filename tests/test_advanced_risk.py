import json

from trustlaya.agent_risk import analyze_agent
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS
from trustlaya.policy import decide
from trustlaya.risk_fusion import fuse
from trustlaya.session_risk import SessionRisk


def low_scores():
    return {task: 0.1 for task in TASKS}


def test_agent_permission_interactions():
    state = {"agent": True, "credential_access": True, "network": True, "human_approval": False}
    agent = analyze_agent(state)
    assert agent["overall_agent_risk"] == 1
    assert decide(low_scores(), [], state, .9, agent_risk=agent) == (
        "BLOCK", "agent_credential_external_channel"
    )
    assert analyze_agent({})["overall_agent_risk"] == 0


def test_fusion_methods_and_interaction():
    scores = low_scores()
    evidence = [{"type": "PII_MENTION", "text": "test", "start": 0, "end": 4},
                {"type": "EXTERNAL_TRANSFER", "text": "send", "start": 5, "end": 9}]
    weighted = fuse(scores, evidence, method="weighted")
    interactive = fuse(scores, evidence, method="interaction")
    constrained = fuse(scores, evidence)
    assert weighted["score"] < interactive["score"] == 0.8
    assert constrained["score"] >= interactive["score"]


def test_session_credential_exfiltration_chain_without_raw_storage():
    tracker = SessionRisk()
    scores = low_scores()
    for text, metadata in [
        ("List files", {"filesystem": True}),
        ("Open configuration", {"shell": True}),
        ("Find credential", {"credential_access": True}),
    ]:
        assessment = tracker.preview(text, scores, metadata=metadata)
        tracker.record(assessment, "REVIEW")
    assessment = tracker.preview("Send credential externally", scores,
                                 metadata={"external_api": True})
    assert "credential_external_transfer_chain" in assessment["triggers"]
    assert decide(scores, [], confidence=.4, session=assessment) == (
        "BLOCK", "session_credential_exfiltration"
    )
    assert all("text" not in row for row in tracker.history)


def test_analyzer_extended_schema_and_session():
    analyzer = Analyzer("onnx")
    session = SessionRisk()
    result = analyzer.analyze("Toplantı notlarını özetle.", session=session)
    json.dumps(result)
    assert set(TASKS) <= result["raw_scores"].keys()
    assert set(TASKS) <= result["calibrated_scores"].keys()
    assert result["risk_fusion"]["source"] == "deterministic_policy_not_probability"
    assert result["session_risk"]["steps_seen"] == 1
    assert len(session.history) == 1
