"""Coverage gate, operation_id idempotency and execution_status, without Docker.

The target is the real demo SQLite API; the analyzer is fake unless stated.
"""

import http.client
import json
import socket
import sys
import threading
from threading import Thread

import pytest

from trustlaya.agent_risk import PERMISSIONS
from trustlaya.api import make_server
from trustlaya.guarded_tool import GuardedTool
from trustlaya.inference import ROOT, Analyzer
from trustlaya.trusted_adapter import make_adapter

sys.path.insert(0, str(ROOT / "demo/firewall"))
from protected_api import dump, make_target, write_record  # noqa: E402

GATEWAY_KEY = "synthetic-gateway-key-1234567890123456"
ADAPTER_KEY = "synthetic-adapter-key-1234567890123456"
TARGET_KEY = "synthetic-target-key-1234567890"
TOOL = {"name": "record", "operation": "write", "target": "demo://records"}
PERMS = {key: key == "database" for key in PERMISSIONS}
RULES = [{**TOOL, "permissions": PERMS, "agents": ["agent-1"]}]
COMPLETE = {"strategy": "head_94_v1", "total_tokens": 5, "read_tokens": 5, "truncated": False}
V2_DIR = ROOT / "models/trustlaya-s-v2"
V2_ONNX = ROOT / "models/exported/v2/trustlaya_s.onnx"


class FakeAnalyzer:
    policy = {}

    def analyze(self, text, metadata=None, session=None):
        evidence, action, coverage = [], "ALLOW", COMPLETE
        if "05551234567" in text:
            start = text.index("05551234567")
            evidence = [{"type": "PHONE", "start": start, "end": start + 11}]
            action = "REDACT"
        if "LONG" in text:
            coverage = {**COMPLETE, "total_tokens": 120, "read_tokens": 94, "truncated": True}
        if "NOCOVERAGE" in text:
            coverage = None
        scores = {key: 0.01 for key in ("pii", "secret", "prompt_injection",
                  "dangerous_instruction", "privacy_risk", "security_risk",
                  "ethics_risk", "oversight_risk", "data_governance_risk")}
        return {**scores, "action": action, "policy_reason": action.lower(),
                "evidence": evidence, "coverage": coverage}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post(port, path, body, headers):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("POST", path, json.dumps(body).encode(),
                           {"Content-Type": "application/json", **headers})
        return json.loads(connection.getresponse().read())
    finally:
        connection.close()


@pytest.fixture
def stack(tmp_path):
    db = tmp_path / "records.sqlite"
    fault = tmp_path / "fault_drop_next_response"
    target = make_target("127.0.0.1", 0, str(db), TARGET_KEY, str(fault))
    gateway_port = free_port()
    adapter = make_adapter(port=0, gateway_url=f"http://127.0.0.1:{gateway_port}",
                           gateway_key=GATEWAY_KEY,
                           adapter_key=ADAPTER_KEY,
                           target_url=f"http://127.0.0.1:{target.server_port}/records",
                           target_key=TARGET_KEY, tool=TOOL, permissions=PERMS)
    gateway = make_server(port=gateway_port, analyzer=FakeAnalyzer(), tool_rules=RULES,
                          shared_key=GATEWAY_KEY,
                          adapter_url=f"http://127.0.0.1:{adapter.server_port}",
                          adapter_key=ADAPTER_KEY, adapter_timeout=5)
    servers = (target, adapter, gateway)
    workers = [Thread(target=s.serve_forever, daemon=True) for s in servers]
    for worker in workers:
        worker.start()
    client = GuardedTool(agent_id="agent-1", session_id="s1", tool=TOOL, permissions=PERMS,
                         gateway_url=f"http://127.0.0.1:{gateway.server_port}",
                         shared_key=GATEWAY_KEY)

    def tool(text, operation_id):
        request = client.build_request(text, {"operation_id": operation_id})
        return post(gateway.server_port, "/v1/tool", request, {"X-TrustLaya-Key": GATEWAY_KEY})

    try:
        yield tool, db, fault, client
    finally:
        for server in servers:
            server.shutdown()
        for server in servers:
            server.server_close()


def test_same_operation_executes_once_and_conflict_is_rejected(stack):
    tool, db, _, _ = stack
    first = tool("Normal note", "op-1")
    retry = tool("Normal note", "op-1")
    conflict = tool("Different note", "op-1")
    assert first["executed"] and first["output"]["replayed"] is False
    assert retry["executed"] and retry["output"] == {**first["output"], "replayed": True}
    assert conflict["executed"] is False and conflict["execution_status"] == "not_executed"
    assert conflict["reason"] == "operation_id_conflict"
    assert [r["text"] for r in dump(str(db))["records"]] == ["Normal note"]


def test_lost_target_reply_is_unknown_and_retry_has_no_second_effect(stack):
    tool, db, fault, _ = stack
    fault.touch()
    lost = tool("Order note", "op-lost")
    assert lost["executed"] is False and lost["execution_status"] == "unknown"
    assert len(dump(str(db))["records"]) == 1  # it did happen
    retry = tool("Order note", "op-lost")
    assert retry["executed"] and retry["output"]["replayed"] is True
    assert dump(str(db))["operations"] == len(dump(str(db))["records"]) == 1


def test_redact_keeps_operation_id_and_sends_only_sanitized_text(stack):
    tool, db, _, _ = stack
    result = tool("Call 05551234567 later", "op-phone")
    assert result["decision"] == "REDACT" and result["executed"]
    records = dump(str(db))["records"]
    assert [(r["operation_id"], r["text"]) for r in records] == [("op-phone", "Call [REDACTED] later")]


@pytest.mark.parametrize("text", ["LONG text", "NOCOVERAGE text"])
def test_incomplete_or_missing_coverage_is_review_and_never_executes(stack, text):
    tool, db, _, _ = stack
    result = tool(text, "op-long")
    assert result["decision"] == "REVIEW" and result["executed"] is False
    assert result["reason_codes"][-1] == "incomplete_analysis_coverage"
    assert result["authorization_issued"] is False
    assert dump(str(db))["records"] == []


def test_invalid_operation_id_is_rejected(stack):
    _, _, _, client = stack
    with pytest.raises(ValueError):
        client.build_request("Normal note", {"operation_id": "bad id with spaces"})


def test_concurrent_consumption_of_one_token_succeeds_once(stack):
    _, _, _, client = stack
    request = client.build_request("Normal note", {"operation_id": "op-race"})
    authorization = client.authorize(request)
    body = {"authorization_token": authorization["authorization_token"],
            "decision_id": authorization["decision_id"],
            "authorized_request": authorization["authorized_request"]}
    results, barrier = [], threading.Barrier(16)

    def consume():
        barrier.wait()
        results.append(client._post("/v1/consume", body)["valid"])

    threads = [Thread(target=consume) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == [False] * 15 + [True]


def test_target_write_is_atomic_under_concurrency(tmp_path):
    db = str(tmp_path / "records.sqlite")
    make_target("127.0.0.1", 0, db, TARGET_KEY).server_close()  # creates schema
    results, barrier = [], threading.Barrier(12)

    def write():
        barrier.wait()
        results.append(write_record(db, "agent-1", "same text", {"operation_id": "op-x"}))

    threads = [Thread(target=write) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(status == 200 for status, _ in results)
    assert sum(not body["replayed"] for _, body in results) == 1
    assert len(dump(db)["records"]) == 1


@pytest.mark.skipif(not V2_ONNX.exists(), reason="V2 ONNX not downloaded")
def test_real_v2_reports_reading_coverage():
    analyzer = Analyzer("onnx", model_dir=V2_DIR, onnx_path=V2_ONNX)
    short = analyzer.analyze("Toplantı notu: proje teslimi cuma günü yapılacak.")
    assert short["coverage"]["truncated"] is False
    assert short["coverage"]["read_tokens"] == short["coverage"]["total_tokens"]
    assert short["versions"]["reading"] == "head_94_v1"
    assert short["versions"]["model"].startswith("sha256:")
    long = analyzer.analyze("Toplantı notu. " * 60)
    assert long["coverage"]["truncated"] is True and long["coverage"]["read_tokens"] == 94
    assert long["coverage"]["unread_spans"]
