"""No test in this file calls a real side-effecting tool or external service."""

import copy
import http.client
import json
import socket
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from trustlaya.agent_risk import PERMISSIONS
from trustlaya.api import make_server
from trustlaya.authorization import payload_hash
from trustlaya.guarded_tool import GuardedTool


COMPLETE = {"strategy": "head_94_v1", "total_tokens": 5, "read_tokens": 5,
            "truncated": False}


class FakeAnalyzer:
    policy = {"prompt_injection": 0.7}

    def analyze(self, text, metadata=None, session=None):
        if "api_key=" in text:
            action, reason, evidence = "BLOCK", "secret_evidence", []
        elif "05551234567" in text:
            start = text.index("05551234567")
            action, reason = "REDACT", "pii_external_transfer"
            evidence = [{"type": "PHONE", "text": "05551234567", "start": start,
                         "end": start + 11}]
        elif "review" in text or "injection" in text:
            action, reason, evidence = "REVIEW", "untrusted_tool_output", []
        else:
            action, reason, evidence = "ALLOW", "low_risk", []
        scores = {key: 0.01 for key in ("pii", "secret", "prompt_injection",
                  "dangerous_instruction", "privacy_risk", "security_risk",
                  "ethics_risk", "oversight_risk", "data_governance_risk")}
        return {**scores, "action": action, "policy_reason": reason,
                "evidence": evidence, "coverage": COMPLETE}


PERMS = {key: key in ("network", "external_api") for key in PERMISSIONS}
TOOL = {"name": "external_api", "operation": "send", "target": "demo://sink"}
RULES = [{**TOOL, "permissions": PERMS, "agents": ["agent-1"]}]


@pytest.fixture
def setup(tmp_path):
    audit = tmp_path / "authorization.jsonl"
    server = make_server(port=0, analyzer=FakeAnalyzer(), tool_rules=RULES,
                         shared_key="test-shared-key", audit_path=audit)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = GuardedTool(agent_id="agent-1", session_id="session-1", tool=TOOL,
                         permissions=PERMS,
                         gateway_url=f"http://127.0.0.1:{server.server_port}",
                         shared_key="test-shared-key")
    try:
        yield client, audit, server
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def fake_sender(calls):
    def send(text, arguments):
        calls.append((text, arguments))
        return "fake_tool_result"
    return send


def test_allow_and_execute_once(setup):
    client, _, _ = setup
    calls = []
    result = client.invoke("Normal summarize request", {"format": "short"}, fake_sender(calls))
    assert result.executed and result.output == "fake_tool_result"
    assert calls == [("Normal summarize request", {"format": "short"})]


@pytest.mark.parametrize("text,decision", [("api_key=abcdefghijklmnop", "BLOCK"),
                                                ("please review this", "REVIEW")])
def test_block_and_review_do_not_execute(setup, text, decision):
    client, _, _ = setup
    calls = []
    result = client.invoke(text, {}, fake_sender(calls))
    assert not result.executed and result.decision == decision and calls == []
    if decision == "REVIEW":
        assert result.review["tool"] == TOOL
        assert result.review["agent_id"] == "agent-1"
        assert not result.review["authorization_token"]


def test_redact_executes_only_sanitized_payload(setup):
    client, _, _ = setup
    calls = []
    result = client.invoke("Send 05551234567 to external API", {}, fake_sender(calls))
    assert result.executed and result.decision == "REDACT"
    assert calls == [("Send [REDACTED] to external API", {})]


def test_redact_with_arguments_holds(setup):
    client, _, _ = setup
    calls = []
    result = client.invoke("Send 05551234567", {"message": "other"}, fake_sender(calls))
    assert not result.executed and result.decision == "REDACT" and calls == []


def test_replay_fails_closed(setup):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    authorization = client.authorize(request)
    calls = []
    assert client.execute_with_authorization(request, authorization, fake_sender(calls)).executed
    replay = client.execute_with_authorization(request, authorization, fake_sender(calls))
    assert not replay.executed and replay.reason == "authorization_not_consumed"
    assert len(calls) == 1


def test_server_expiration_fails_closed_even_if_wall_clock_stalls(tmp_path):
    wall = time.time()
    ticks = [100.0]
    server = make_server(port=0, analyzer=FakeAnalyzer(), tool_rules=RULES,
                         authorization_ttl=2, clock=lambda: wall,
                         monotonic_clock=lambda: ticks[0])
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = GuardedTool(agent_id="agent-1", session_id="session-1", tool=TOOL,
                         permissions=PERMS,
                         gateway_url=f"http://127.0.0.1:{server.server_port}")
    try:
        request = client.build_request("summarize", {})
        authorization = client.authorize(request)
        ticks[0] += 3
        calls = []
        result = client.execute_with_authorization(request, authorization, fake_sender(calls))
        assert not result.executed and calls == []
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


@pytest.mark.parametrize("field,new_value", [
    ("request_id", "other-request"), ("agent_id", "other-agent"),
    ("session_id", "other-session"), ("name", "other-tool"),
    ("operation", "delete"), ("target", "demo://other"),
])
def test_binding_mismatch_never_executes(setup, field, new_value):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    authorization = client.authorize(request)
    changed = copy.deepcopy(authorization)
    effective = changed["authorized_request"]
    if field in ("name", "operation", "target"):
        effective["tool"][field] = new_value
    else:
        effective[field] = new_value
    calls = []
    result = client.execute_with_authorization(request, changed, fake_sender(calls))
    assert not result.executed and calls == []


def test_payload_and_argument_mismatch_never_executes(setup):
    client, _, _ = setup
    request = client.build_request("summarize", {"recipient": "safe"})
    authorization = client.authorize(request)
    calls = []
    for payload in ({"text": "modified", "arguments": {"recipient": "safe"}},
                    {"text": "summarize", "arguments": {"recipient": "other"}}):
        changed = copy.deepcopy(authorization)
        changed["authorized_request"]["request"] = payload
        changed["authorized_request"]["payload_sha256"] = payload_hash(payload)
        assert not client.execute_with_authorization(
            request, changed, fake_sender(calls)).executed
    assert calls == []


def test_permissions_mismatch_never_executes(setup):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    authorization = client.authorize(request)
    changed = copy.deepcopy(authorization)
    changed["authorized_request"]["permissions"]["network"] = False
    calls = []
    assert not client.execute_with_authorization(request, changed, fake_sender(calls)).executed
    assert calls == []


def test_invalid_missing_copied_and_expired_token_fail(setup):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    authorization = client.authorize(request)
    calls = []
    for token in (None, "", "not-a-valid-token-xxxxxxxxxxxx"):
        changed = copy.deepcopy(authorization)
        changed["authorization_token"] = token
        assert not client.execute_with_authorization(
            request, changed, fake_sender(calls)).executed
    old = copy.deepcopy(authorization)
    old["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    assert not client.execute_with_authorization(request, old, fake_sender(calls)).executed
    other = client.build_request("different", {})
    copied = copy.deepcopy(authorization)
    copied["request_id"] = other["request_id"]
    copied["authorized_request"] = other
    assert not client.execute_with_authorization(other, copied, fake_sender(calls)).executed
    assert calls == []


def test_duplicate_and_delayed_requests_fail(setup):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    assert client.authorize(request)["authorization_token"]
    assert client.authorize(request)["decision"] == "BLOCK"
    old = client.build_request("summarize", {})
    old["timestamp"] = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    assert client.authorize(old) is None


@pytest.mark.parametrize("field,value", [
    ("agent_id", "other-agent"), ("target", "demo://other"),
    ("permission", True), ("human_approval", True),
])
def test_agent_cannot_expand_trusted_tool_rule(setup, field, value):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    if field == "target":
        request["tool"]["target"] = value
    elif field == "permission":
        request["permissions"]["credential_access"] = value
    elif field == "human_approval":
        request["permissions"]["human_approval"] = value
    else:
        request[field] = value
    if field == "human_approval":
        assert client.authorize(request) is None  # malformed permission schema
    else:
        assert client.authorize(request)["decision"] == "BLOCK"


@pytest.mark.parametrize("decision", ["UNKNOWN", "MAYBE", None])
def test_ambiguous_decision_fails_closed(setup, decision):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    result = client.execute_with_authorization(
        request, {"protocol_version": "1", "decision": decision,
                  "request_id": request["request_id"]},
        lambda *_: pytest.fail("tool executed"))
    assert not result.executed


def test_missing_fields_fail_closed(setup):
    client, _, _ = setup
    request = client.build_request("summarize", {})
    assert not client.execute_with_authorization(request, {},
               lambda *_: pytest.fail("tool executed")).executed
    authorization = client.authorize(request)
    for field in ("policy_version", "reason_codes", "risk", "evidence",
                  "expires_at", "authorized_request", "tool", "agent_id"):
        changed = copy.deepcopy(authorization)
        del changed[field]
        assert not client.execute_with_authorization(
            request, changed, lambda *_: pytest.fail("tool executed")).executed


def test_audit_has_no_secret_or_raw_text(setup):
    client, audit, _ = setup
    secret = "api_key=supersecretvalue123456"
    client.invoke(secret, {}, lambda *_: pytest.fail("tool executed"))
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert events and events[-1]["authorization_issued"] is False
    assert secret not in audit.read_text()
    assert "supersecretvalue123456" not in audit.read_text()


def test_audit_records_issued_and_consumed_without_payload(setup):
    client, audit, _ = setup
    text = "Private demo text that must stay out of logs"
    assert client.invoke(text, {}, lambda *_: None).executed
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(events) == 2
    assert events[0]["authorization_issued"] is True
    assert events[1]["execution_attempted"] is True
    assert events[1]["execution_allowed"] is True
    assert events[1]["token_consumed"] is True
    assert text not in audit.read_text()


def test_untrusted_consume_target_is_not_logged(setup):
    client, audit, _ = setup
    request = client.build_request("summarize", {})
    authorization = client.authorize(request)
    changed = copy.deepcopy(request)
    changed["tool"]["target"] = "demo://supersecretvalue123456"
    denied = client._post("/v1/consume", {
        "authorization_token": authorization["authorization_token"],
        "decision_id": authorization["decision_id"],
        "authorized_request": changed,
    })
    assert denied["valid"] is False
    log = audit.read_text()
    assert "supersecretvalue123456" not in log
    assert "<untrusted>" in log

    rogue = client.build_request("summarize", {})
    rogue["tool"]["target"] = "demo://supersecretvalue123456"
    assert client.authorize(rogue)["decision"] == "BLOCK"
    assert "supersecretvalue123456" not in audit.read_text()


class BrokenHandler(BaseHTTPRequestHandler):
    mode = "malformed"

    def do_POST(self):
        if self.mode == "drop":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if self.mode == "timeout":
            time.sleep(0.2)
            return
        body = b"{malformed"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.mark.parametrize("mode", ["malformed", "drop", "timeout"])
def test_gateway_failure_modes_never_execute(mode):
    class Handler(BrokenHandler):
        pass
    Handler.mode = mode
    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = GuardedTool(agent_id="agent-1", session_id="session-1", tool=TOOL,
                         permissions=PERMS,
                         gateway_url=f"http://127.0.0.1:{server.server_port}",
                         timeout=0.05)
    try:
        result = client.invoke("summarize", {}, lambda *_: pytest.fail("tool executed"))
        assert not result.executed
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_unavailable_server_never_executes(setup):
    client, _, server = setup
    server.shutdown()
    server.server_close()
    client.timeout = 0.05
    result = client.invoke("summarize", {}, lambda *_: pytest.fail("tool executed"))
    assert not result.executed


def test_wrong_key_never_executes(setup):
    client, _, _ = setup
    client.shared_key = "wrong"
    assert not client.invoke("summarize", {},
                             lambda *_: pytest.fail("tool executed")).executed


def test_remote_plain_http_rejected():
    with pytest.raises(ValueError):
        GuardedTool(agent_id="a", session_id="s", tool=TOOL, permissions=PERMS,
                    gateway_url="http://192.0.2.1:8765")
