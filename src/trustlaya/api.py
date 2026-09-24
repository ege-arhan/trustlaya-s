"""Local, dependency-free HTTP gateway for edge clients such as UNO Q."""

import json
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .inference import Analyzer
from .session_risk import SessionRisk
from .authorization import ProtocolError
from .authorization_service import AuthorizationService
import secrets

DETECTIONS = ("pii", "secret", "prompt_injection", "dangerous_instruction")
RISKS = ("privacy_risk", "security_risk", "ethics_risk", "oversight_risk",
         "data_governance_risk")


def public_result(result):
    return {
        "detections": {key: result[key] for key in DETECTIONS},
        "risks": {**{key: result[key] for key in RISKS},
                  "agent": result["agent_risk"],
                  "session": result["session_risk"],
                  "fusion": result["risk_fusion"]},
        "raw_scores": result["raw_scores"],
        "calibrated_scores": result["calibrated_scores"],
        "confidence": result["confidence"], "abstain": result["abstain"],
        "evidence": result["evidence"], "severity": result["severity"],
        "model_action": result["model_action"], "action": result["action"],
        "policy_rule": result["policy_reason"],
        "timing_ms": result.get("timing_ms"),
    }


def audit_record(result):
    """Never store text, credential values, evidence text, or raw prompts."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk": {key: round(float(result[key]), 4) for key in DETECTIONS + RISKS},
        "action": result["action"], "policy_rule": result["policy_reason"],
        "evidence": [{"type": item["type"], "start": item["start"], "end": item["end"]}
                     for item in result["evidence"]],
    }


def make_server(host="127.0.0.1", port=8765, analyzer=None, audit_path=None,
                tool_rules=None, shared_key=None, authorization_ttl=15, clock=None,
                monotonic_clock=None):
    if host not in ("127.0.0.1", "localhost", "::1") and (
            not isinstance(shared_key, str) or len(shared_key) < 32):
        raise ValueError("non-loopback binding requires a 32+ character key and TLS proxy")
    engine = analyzer or Analyzer("onnx")
    sessions = {}
    audit_path = Path(audit_path) if audit_path else None
    authority = (AuthorizationService(engine, tool_rules, ttl_seconds=authorization_ttl,
                                      clock=clock, monotonic_clock=monotonic_clock)
                 if tool_rules is not None else None)

    def write_audit(event):
        if audit_path is not None and event is not None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a") as out:
                out.write(json.dumps(event) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path in ("/v1/authorize", "/v1/consume"):
                self.authorization_route()
                return
            if self.path != "/analyze":
                self.respond(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("body size must be 1..65536 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object required")
                text = payload.get("text")
                if not isinstance(text, str) or not 0 < len(text) <= 2000:
                    raise ValueError("text must be 1..2000 characters")
                metadata = payload.get("agent_state") or {}
                if not isinstance(metadata, dict) or any(not isinstance(v, bool) for v in metadata.values()):
                    raise ValueError("agent_state must contain boolean values")
                if "policy" in payload:
                    raise ValueError("policy overrides are not accepted by the public API")
                session_id = payload.get("session_id")
                if session_id is not None and (not isinstance(session_id, str) or
                                               not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id)):
                    raise ValueError("invalid session_id")
                tracker = None
                if session_id is not None:
                    if session_id not in sessions and len(sessions) >= 128:
                        sessions.pop(next(iter(sessions)))
                    tracker = sessions.setdefault(session_id, SessionRisk())
                result = engine.analyze(text, metadata, session=tracker)
                write_audit(audit_record(result))
                self.respond(200, public_result(result))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})

        def authorization_route(self):
            if authority is None:
                self.respond(503, {"error": "authorization_not_configured"})
                return
            if shared_key and not secrets.compare_digest(
                    self.headers.get("X-TrustLaya-Key", ""), shared_key):
                self.respond(401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ProtocolError("invalid_body_size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ProtocolError("invalid_json_body")
                if self.path == "/v1/authorize":
                    request = payload
                    session_key = (request.get("agent_id"), request.get("session_id"))
                    tracker = sessions.get(session_key)
                    if tracker is None and all(isinstance(x, str) for x in session_key):
                        if len(sessions) >= 128:
                            sessions.pop(next(iter(sessions)))
                        tracker = sessions.setdefault(session_key, SessionRisk())
                    response, event = authority.authorize(request, session=tracker)
                else:
                    if set(payload) != {"authorization_token", "authorized_request", "decision_id"}:
                        raise ProtocolError("invalid_consume_schema")
                    response, event = authority.consume(payload["authorization_token"],
                                                        payload["authorized_request"],
                                                        payload["decision_id"])
                write_audit(event)
                self.respond(200, response)
            except (ProtocolError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception:
                # A model, audit or storage failure must never issue authorization.
                self.respond(503, {"error": "authorization_unavailable"})

        def respond(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            # HTTP access logs are disabled; optional audit_record contains no raw input.
            pass

    return HTTPServer((host, port), Handler)
