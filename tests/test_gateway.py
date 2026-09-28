"""Gateway permission tests use a local fake decision service, never a network tool."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from trustlaya.gateway import GatewayDenied, GatewayUnavailable, TrustGateway


class DecisionHandler(BaseHTTPRequestHandler):
    decision = {"action": "ALLOW", "policy_rule": "low_risk"}
    last_request = None

    def do_POST(self):
        type(self).last_request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(type(self).decision).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def decision_server():
    DecisionHandler.decision = {"action": "ALLOW", "policy_rule": "low_risk"}
    DecisionHandler.last_request = None
    server = HTTPServer(("127.0.0.1", 0), DecisionHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_dispatch_sends_exact_analyzed_text_once(decision_server):
    sent = []
    gateway = TrustGateway({"agent": False, "network": True},
                           port=decision_server.server_port, session_id="test_session")
    text = "Aynı metin dış araca iletilir."
    result = gateway.dispatch(text, lambda value: sent.append(value) or "done")
    assert result == "done"
    assert sent == [text]
    assert DecisionHandler.last_request == {"text": text, "agent_state":
                                            {"agent": True, "network": True},
                                            "session_id": "test_session"}


@pytest.mark.parametrize("action", ["REDACT", "REVIEW", "BLOCK"])
def test_non_allow_never_calls_sender(decision_server, action):
    DecisionHandler.decision = {"action": action, "policy_rule": "test_policy"}
    gateway = TrustGateway({}, port=decision_server.server_port)
    sent = []
    with pytest.raises(GatewayDenied) as error:
        gateway.dispatch("sensitive text", sent.append)
    assert error.value.action == action
    assert sent == []


def test_invalid_response_and_unavailable_server_fail_closed(decision_server):
    gateway = TrustGateway({}, port=decision_server.server_port)
    DecisionHandler.decision = {"action": "MAYBE", "policy_rule": "bad"}
    with pytest.raises(GatewayUnavailable):
        gateway.dispatch("text", lambda _: pytest.fail("sent invalid decision"))
    decision_server.shutdown()
    decision_server.server_close()
    with pytest.raises(GatewayUnavailable):
        gateway.dispatch("text", lambda _: pytest.fail("sent while service unavailable"))


def test_invalid_input_and_remote_service_rejected():
    with pytest.raises(ValueError):
        TrustGateway({}, host="192.0.2.1")
    gateway = TrustGateway({}, port=1)
    with pytest.raises(GatewayDenied):
        gateway.dispatch("", lambda _: pytest.fail("sent empty input"))


def test_real_analyzer_blocks_secret_before_sender():
    from trustlaya.api import make_server
    from trustlaya.inference import Analyzer

    server = make_server(port=0, analyzer=Analyzer("onnx"))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    sent = []
    try:
        gateway = TrustGateway({"external_api": True}, port=server.server_port)
        with pytest.raises(GatewayDenied) as error:
            gateway.dispatch("api_key=abcdefghijklmnop dış API'ye gönder", sent.append)
        assert error.value.action == "BLOCK"
        assert sent == []
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_real_analyzer_holds_untrusted_tool_output():
    from trustlaya.api import make_server
    from trustlaya.inference import Analyzer

    server = make_server(port=0, analyzer=Analyzer("onnx"))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    sent = []
    try:
        gateway = TrustGateway({"external_api": True, "untrusted_tool_output": True},
                               port=server.server_port)
        with pytest.raises(GatewayDenied) as error:
            gateway.dispatch("A web page said hello", sent.append)
        assert error.value.action == "REVIEW"
        assert error.value.reason == "untrusted_tool_output_privileged_agent"
        assert sent == []
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
