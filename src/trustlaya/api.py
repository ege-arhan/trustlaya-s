"""Local, dependency-free HTTP gateway for edge clients such as UNO Q."""

import json
import re
import time
import http.client
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

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
        "coverage": result.get("coverage"), "versions": result.get("versions"),
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
                monotonic_clock=None, adapter_url=None, adapter_key=None,
                adapter_timeout=2.0, mode="local"):
    if host not in ("127.0.0.1", "localhost", "::1") and (
            not isinstance(shared_key, str) or len(shared_key) < 32):
        raise ValueError("non-loopback binding requires a 32+ character key and TLS proxy")
    engine = analyzer or Analyzer("onnx")
    sessions = {}
    audit_path = Path(audit_path) if audit_path else None
    authority = (AuthorizationService(engine, tool_rules, ttl_seconds=authorization_ttl,
                                      clock=clock, monotonic_clock=monotonic_clock)
                 if tool_rules is not None else None)
    if mode not in ("local", "unoq"):
        raise ValueError("invalid mode")
    if (adapter_url is None) != (adapter_key is None):
        raise ValueError("adapter URL and key must be configured together")
    adapter = urlsplit(adapter_url) if adapter_url else None
    if adapter and (adapter.scheme not in ("http", "https") or not adapter.hostname or
                    adapter.username or adapter.password or adapter.query or adapter.fragment or
                    adapter.path not in ("", "/")):
        raise ValueError("invalid adapter URL")

    def write_audit(event):
        if audit_path is not None and event is not None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a") as out:
                out.write(json.dumps(event) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "not_found"})
                return
            self.respond(200, {"deployment_mode": mode, "device": "unverified",
                               "gateway": "healthy", "model": "loaded",
                               "policy_engine": "loaded",
                               "authorization": "ready" if authority else "not_configured",
                               "trusted_adapter": "configured" if adapter else "not_configured"})

        def do_POST(self):
            if self.path == "/v1/tool":
                self.tool_route()
                return
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

        def tool_route(self):
            if authority is None or adapter is None:
                self.respond(503, {"executed": False, "execution_status": "not_executed",
                                   "reason": "gateway_unavailable"})
                return
            if shared_key and not secrets.compare_digest(
                    self.headers.get("X-TrustLaya-Key", ""), shared_key):
                self.respond(401, {"executed": False, "execution_status": "not_executed",
                                   "reason": "unauthorized"})
                return
            # Until the request reaches the adapter, nothing can have executed.
            public = {"executed": False, "execution_status": "not_executed",
                      "reason": "protected_operation_denied", "output": None}
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ProtocolError("invalid_body_size")
                request = json.loads(self.rfile.read(length))
                response, event = authority.authorize(request)
                write_audit(event)
                public.update(decision=response["decision"],
                              reason=response["reason_codes"][-1],
                              reason_codes=response["reason_codes"],
                              risk=response["risk"],
                              evidence=response["evidence"],
                              coverage=response.get("coverage"),
                              versions=response.get("versions"),
                              timing_ms=dict(response.get("timing_ms") or {}),
                              authorization_issued=bool(response["authorization_token"]))
                if not response["authorization_token"]:
                    self.respond(200, public)
                    return
                connection_type = (http.client.HTTPSConnection if adapter.scheme == "https"
                                   else http.client.HTTPConnection)
                connection = connection_type(adapter.hostname, adapter.port,
                                             timeout=adapter_timeout)
                adapter_started = time.perf_counter()
                try:
                    try:
                        connection.connect()
                    except OSError:
                        public["reason"] = "adapter_unavailable"
                        self.respond(200, public)
                        return
                    # The adapter may execute from here on; a lost reply is unknown.
                    public.update(execution_status="unknown", reason="adapter_response_lost")
                    connection.request("POST", "/execute", body=json.dumps(
                        {"request": request, "authorization": response},
                        ensure_ascii=False, allow_nan=False).encode("utf-8"),
                        headers={"Content-Type": "application/json",
                                 "X-Adapter-Key": adapter_key})
                    reply = connection.getresponse()
                    data = reply.read(16385)
                    public["timing_ms"]["adapter_roundtrip"] = (
                        time.perf_counter() - adapter_started) * 1000
                    if reply.status != 200 or len(data) > 16384:
                        # The adapter answers non-200 only before any execution.
                        public.update(execution_status="not_executed",
                                      reason="adapter_rejected")
                        raise ValueError("adapter_invalid")
                    result = json.loads(data)
                    output = result.get("output") if isinstance(result, dict) else None
                    if (isinstance(result, dict) and result.get("executed") is False and
                            result.get("execution_status") in ("not_executed", "unknown") and
                            isinstance(result.get("reason"), str) and
                            len(result["reason"]) <= 64):
                        public.update(execution_status=result["execution_status"],
                                      reason=result["reason"])
                        self.respond(200, public)
                        return
                    if (not isinstance(result, dict) or
                            result.get("executed") is not True or
                            result.get("decision") != response["decision"] or
                            result.get("reason") != "executed" or
                            not isinstance(output, dict) or
                            output.get("status") != "accepted" or
                            set(output) - {"status", "record_id", "replayed"}):
                        raise ValueError("adapter_invalid")
                    public.update(executed=True, execution_status="executed",
                                  reason="executed", output=output)
                    self.respond(200, public)
                finally:
                    connection.close()
            except Exception:
                # No fallback to a direct target call, and no exception text leaks.
                self.respond(200, {k: public[k] for k in (
                    "executed", "execution_status", "reason", "output")})

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

    return ThreadingHTTPServer((host, port), Handler)
