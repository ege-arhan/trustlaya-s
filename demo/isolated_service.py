"""Deterministic container harness; no model claim and no real API calls."""

import json
import os
import sys
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeAnalyzer:
    policy = {"secret": 0.8}

    def analyze(self, text, metadata=None, session=None):
        decision = ("BLOCK" if "api_key=" in text else
                    "REVIEW" if "privileged shell" in text else
                    "REDACT" if "05551234567" in text else "ALLOW")
        evidence = []
        if decision == "REDACT":
            start = text.index("05551234567")
            evidence = [{"type": "PHONE", "start": start, "end": start + 11}]
        keys = ("pii", "secret", "prompt_injection", "dangerous_instruction",
                "privacy_risk", "security_risk", "ethics_risk", "oversight_risk",
                "data_governance_risk")
        return {**{key: 0.01 for key in keys}, "action": decision,
                "policy_reason": decision.lower(), "evidence": evidence}


def target():
    expected = os.environ["TRUSTEDGE_TARGET_KEY"]
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/count":
                self.send_response(404)
                self.end_headers()
                return
            data = json.dumps({"count": len(received), "texts": received}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.headers.get("X-Target-Key") != expected:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            received.append(json.loads(body)["text"])
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            pass

    ThreadingHTTPServer(("0.0.0.0", 8767), Handler).serve_forever()


def gateway():
    # Keep this container harness stdlib-only. The production server imports
    # the real Analyzer; these mocked policy decisions test network isolation.
    sys.modules["trustlaya.inference"] = types.SimpleNamespace(Analyzer=FakeAnalyzer)
    from trustlaya.api import make_server

    rules = json.load(open("/app/configs/guarded_tools.isolated.json"))["tools"]
    make_server("0.0.0.0", 8765, FakeAnalyzer(),
                tool_rules=rules, shared_key=os.environ["TRUSTLAYA_SHARED_KEY"],
                adapter_url="http://adapter:8766",
                adapter_key=os.environ["TRUSTEDGE_ADAPTER_KEY"]).serve_forever()


def adapter():
    from trustlaya.trusted_adapter import make_adapter

    rule = json.load(open("/app/configs/guarded_tools.isolated.json"))["tools"][0]
    make_adapter("0.0.0.0", 8766, gateway_url="http://gateway:8765",
                 gateway_key=os.environ["TRUSTLAYA_SHARED_KEY"],
                 adapter_key=os.environ["TRUSTEDGE_ADAPTER_KEY"],
                 target_url="http://target:8767/send",
                 target_key=os.environ["TRUSTEDGE_TARGET_KEY"],
                 tool={key: rule[key] for key in ("name", "operation", "target")},
                 permissions=rule["permissions"]).serve_forever()


if __name__ == "__main__":
    {"target": target, "gateway": gateway, "adapter": adapter}[sys.argv[1]]()
