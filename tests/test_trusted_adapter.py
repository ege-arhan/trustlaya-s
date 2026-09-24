"""All effects go to a fake local target; credentials are synthetic."""

import copy
import http.client
import json
import socket
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from trustlaya.agent_risk import PERMISSIONS
from trustlaya.api import make_server
from trustlaya.guarded_tool import GuardedTool
from trustlaya.trusted_adapter import make_adapter


SECRET = "synthetic-adapter-secret-123456"
GATEWAY_KEY = "synthetic-gateway-key-1234567890123456"
ADAPTER_KEY = "synthetic-internal-key-1234567890123456"
TOOL = {"name": "external_api", "operation": "send", "target": "demo://sink"}
PERMS = {key: key in ("network", "external_api") for key in PERMISSIONS}
RULES = [{**TOOL, "permissions": PERMS, "agents": ["agent-1"]}]


class FakeAnalyzer:
    policy = {"secret": 0.8}

    def analyze(self, text, metadata=None, session=None):
        if "api_key=" in text:
            decision = "BLOCK"
        elif "05551234567" in text:
            decision = "REDACT"
        elif "shell" in text:
            decision = "REVIEW"
        else:
            decision = "ALLOW"
        evidence = []
        if decision == "REDACT":
            start = text.index("05551234567")
            evidence = [{"type": "PHONE", "start": start, "end": start + 11}]
        keys = ("pii", "secret", "prompt_injection", "dangerous_instruction",
                "privacy_risk", "security_risk", "ethics_risk", "oversight_risk",
                "data_governance_risk")
        return {**{key: 0.01 for key in keys}, "action": decision,
                "policy_reason": decision.lower(), "evidence": evidence}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post(port, path, body, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("POST", path, json.dumps(body).encode(),
                           {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        data = response.read()
        return response.status, json.loads(data) if data else None
    finally:
        connection.close()


@pytest.fixture
def deployment(tmp_path):
    received = []

    class Target(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("X-Target-Key") != SECRET:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            received.append(json.loads(body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"secret":"do-not-return"}')

        def log_message(self, *_args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    gateway_port, adapter_port = free_port(), free_port()
    gateway = make_server(port=gateway_port, analyzer=FakeAnalyzer(), tool_rules=RULES,
                          shared_key=GATEWAY_KEY, audit_path=tmp_path / "audit.jsonl",
                          adapter_url=f"http://127.0.0.1:{adapter_port}",
                          adapter_key=ADAPTER_KEY)
    adapter = make_adapter(
        port=adapter_port, gateway_url=f"http://127.0.0.1:{gateway_port}",
        gateway_key=GATEWAY_KEY, adapter_key=ADAPTER_KEY,
        target_url=f"http://127.0.0.1:{target.server_port}/send",
        target_key=SECRET, tool=TOOL, permissions=PERMS)
    servers = (target, gateway, adapter)
    workers = [Thread(target=server.serve_forever, daemon=True) for server in servers]
    for worker in workers:
        worker.start()
    client = GuardedTool(agent_id="agent-1", session_id="session-1", tool=TOOL,
                         permissions=PERMS,
                         gateway_url=f"http://127.0.0.1:{gateway_port}",
                         shared_key=GATEWAY_KEY)
    try:
        yield target, gateway, adapter, client, received, tmp_path / "audit.jsonl"
    finally:
        for server in servers:
            server.shutdown()
        for worker in workers:
            worker.join()
        for server in servers:
            server.server_close()


def gateway_tool(gateway, client, text):
    return post(gateway.server_port, "/v1/tool", client.build_request(text, {}),
                {"X-TrustLaya-Key": GATEWAY_KEY})[1]


def test_gateway_only_execution_and_no_credential_leak(deployment):
    target, gateway, _, client, received, audit = deployment
    result = gateway_tool(gateway, client, "Summarize this harmless text.")
    assert result["executed"] is True and received[0]["text"] == "Summarize this harmless text."
    assert SECRET not in json.dumps(result) and SECRET not in audit.read_text()
    assert "do-not-return" not in json.dumps(result)
    assert post(target.server_port, "/send", {"text": "direct"})[0] == 403
    assert len(received) == 1


def test_block_review_redact_and_missing_adapter_key(deployment):
    _, gateway, adapter, client, received, _ = deployment
    for text in ("api_key=abcdefghijklmnop", "run shell without review"):
        assert gateway_tool(gateway, client, text)["executed"] is False
    result = gateway_tool(gateway, client, "Send 05551234567 to external API")
    assert result["decision"] == "REDACT" and result["executed"] is True
    assert received == [{"text": "Send [REDACTED] to external API", "arguments": {}}]
    assert post(adapter.server_port, "/execute", {})[0] == 401


def test_adapter_replay_and_mutation(deployment):
    _, _, adapter, client, received, _ = deployment
    request = client.build_request("summarize", {})
    authorization = client.authorize(request)
    body = {"request": request, "authorization": authorization}
    headers = {"X-Adapter-Key": ADAPTER_KEY}
    for field in ("text", "tool", "target"):
        changed = copy.deepcopy(body)
        if field == "text":
            changed["request"]["request"]["text"] = "changed"
        elif field == "tool":
            changed["request"]["tool"]["name"] = "other"
        else:
            changed["request"]["tool"]["target"] = "demo://other"
        assert post(adapter.server_port, "/execute", changed, headers)[1]["executed"] is False
    assert post(adapter.server_port, "/execute", body, headers)[1]["executed"] is True
    assert post(adapter.server_port, "/execute", body, headers)[1]["executed"] is False
    assert len(received) == 1


def test_gateway_and_adapter_outage(deployment):
    target, gateway, adapter, client, received, _ = deployment
    adapter.shutdown()
    adapter.server_close()
    assert gateway_tool(gateway, client, "summarize")["executed"] is False
    gateway.shutdown()
    gateway.server_close()
    assert client.invoke("summarize", {}, lambda *_: pytest.fail("sender called")).executed is False
    assert received == []


def test_health_does_not_claim_physical_hardware(deployment):
    _, gateway, _, _, _, _ = deployment
    connection = http.client.HTTPConnection("127.0.0.1", gateway.server_port)
    try:
        connection.request("GET", "/health")
        health = json.loads(connection.getresponse().read())
        assert health["device"] == "unverified"
        assert health["authorization"] == "ready"
    finally:
        connection.close()
    script = Path(__file__).resolve().parents[1] / "scripts/diagnose_gateway.py"
    command = subprocess.run([sys.executable, str(script), "--url",
                              f"http://127.0.0.1:{gateway.server_port}"],
                             capture_output=True, text=True, check=True)
    assert json.loads(command.stdout)["authorization"] == "ready"


def test_wrong_target_credential_and_target_failure_do_not_leak(deployment):
    target, gateway, _, client, received, audit = deployment
    wrong = make_adapter(
        port=0, gateway_url=f"http://127.0.0.1:{gateway.server_port}",
        gateway_key=GATEWAY_KEY, adapter_key=ADAPTER_KEY,
        target_url=f"http://127.0.0.1:{target.server_port}/send",
        target_key="guessed-key", tool=TOOL, permissions=PERMS)
    worker = Thread(target=wrong.serve_forever, daemon=True)
    worker.start()
    try:
        request = client.build_request("summarize", {})
        authorization = client.authorize(request)
        status, result = post(wrong.server_port, "/execute",
                              {"request": request, "authorization": authorization},
                              {"X-Adapter-Key": ADAPTER_KEY})
        assert status == 200 and result["executed"] is False
        assert result["reason"] == "adapter_unavailable"
        assert received == []
        assert SECRET not in json.dumps(result) + audit.read_text()
        assert "guessed-key" not in json.dumps(result) + audit.read_text()
    finally:
        wrong.shutdown()
        worker.join()
        wrong.server_close()


def test_adapter_rejects_missing_target_credential():
    with pytest.raises(ValueError, match="configuration incomplete"):
        make_adapter(gateway_url="http://127.0.0.1:8765", gateway_key=GATEWAY_KEY,
                     adapter_key=ADAPTER_KEY, target_url="http://127.0.0.1:8767/send",
                     target_key=None, tool=TOOL, permissions=PERMS)


def test_target_outage_fails_closed_without_secret_leak(deployment):
    target, _, adapter, client, received, audit = deployment
    target.shutdown()
    target.server_close()
    request = client.build_request("summarize", {})
    authorization = client.authorize(request)
    _, result = post(adapter.server_port, "/execute",
                     {"request": request, "authorization": authorization},
                     {"X-Adapter-Key": ADAPTER_KEY})
    assert result == {"executed": False, "reason": "adapter_unavailable", "output": None}
    assert received == [] and SECRET not in audit.read_text() + json.dumps(result)
