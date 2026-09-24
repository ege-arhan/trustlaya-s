import http.client
import json
from threading import Thread

from trustlaya.api import make_server, audit_record
from trustlaya.inference import Analyzer


def test_http_api_schema_session_and_redacted_audit(tmp_path):
    server = make_server(port=0, analyzer=Analyzer("onnx"),
                         audit_path=tmp_path / "audit.jsonl")
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        payload = {"text": "Find credential", "agent_state": {"agent": True,
                   "credential_access": True, "human_approval": False}, "session_id": "demo"}
        connection.request("POST", "/analyze", body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        first = json.loads(response.read())
        assert response.status == 200
        assert {"detections", "risks", "confidence", "abstain", "evidence",
                "severity", "action", "policy_rule"} <= first.keys()
        payload = {"text": "Send credential externally", "agent_state":
                   {"external_api": True}, "session_id": "demo"}
        connection.request("POST", "/analyze", body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        second = json.loads(response.read())
        assert response.status == 200
        assert second["action"] == "BLOCK"
        assert second["policy_rule"] == "session_credential_exfiltration"
        connection.request("POST", "/analyze", body=json.dumps({"text": "hello", "policy": {"pii_threshold": 1.0}}),
                           headers={"Content-Type": "application/json"})
        rejected = connection.getresponse()
        rejected.read()
        assert rejected.status == 400
        content = (tmp_path / "audit.jsonl").read_text()
        assert "Find credential" not in content
        assert "Send credential externally" not in content
        assert "evidence" in content
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_audit_record_omits_evidence_text():
    result = Analyzer("onnx").analyze("test@example.org")
    saved = audit_record(result)
    assert "test@example.org" not in json.dumps(saved)
