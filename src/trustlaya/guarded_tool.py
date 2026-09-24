"""Trusted tool-side wrapper: consume gateway authorization before side effects."""

import http.client
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .authorization import VERSION, payload_hash, validate_request


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool
    reason: str
    decision: str | None = None
    output: object = None
    review: dict | None = None
    timing_ms: dict | None = None


class GuardedTool:
    """The sender must exist only in trusted code with direct access restricted."""

    def __init__(self, *, agent_id, session_id, tool, permissions,
                 gateway_url=None, shared_key=None, timeout=2.0):
        self.gateway_url = gateway_url or os.getenv(
            "TRUSTLAYA_GATEWAY_URL", "http://127.0.0.1:8765")
        parsed = urlsplit(self.gateway_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid gateway URL") from exc
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                or parsed.username or parsed.password or not isinstance(timeout, (int, float))
                or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("invalid gateway URL")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("remote gateway requires HTTPS")
        self.url = parsed
        self.port = port
        self.shared_key = shared_key if shared_key is not None else os.getenv("TRUSTLAYA_SHARED_KEY")
        self.timeout = timeout
        self.agent_id = agent_id
        self.session_id = session_id
        self.tool = dict(tool)
        self.permissions = dict(permissions)
        self.timeout_count = 0
        self.denied_execution_count = 0
        self.last_authorization_ms = None

    def build_request(self, text, arguments=None, request_id=None, timestamp=None):
        payload = {"text": text, "arguments": json.loads(json.dumps({} if arguments is None else arguments,
                                                            ensure_ascii=False, allow_nan=False))}
        request = {"protocol_version": VERSION,
                   "request_id": request_id or uuid.uuid4().hex,
                   "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
                   "agent_id": self.agent_id, "session_id": self.session_id,
                   "tool": self.tool.copy(), "permissions": self.permissions.copy(),
                   "request": payload, "payload_sha256": payload_hash(payload)}
        validate_request(request)
        return request

    def _post(self, path, payload):
        connection_type = (http.client.HTTPSConnection if self.url.scheme == "https"
                           else http.client.HTTPConnection)
        connection = None
        headers = {"Content-Type": "application/json"}
        if self.shared_key:
            headers["X-TrustLaya-Key"] = self.shared_key
        try:
            connection = connection_type(self.url.hostname, self.port, timeout=self.timeout)
            connection.request("POST", path, body=json.dumps(
                payload, ensure_ascii=False, allow_nan=False).encode("utf-8"), headers=headers)
            response = connection.getresponse()
            data = response.read(16385)
            if response.status != 200 or len(data) > 16384:
                return None
            return json.loads(data)
        except TimeoutError:
            self.timeout_count += 1
            return None
        except (OSError, ValueError, http.client.HTTPException):
            return None
        finally:
            if connection is not None:
                connection.close()

    def authorize(self, request):
        started = time.perf_counter()
        response = self._post("/v1/authorize", request)
        self.last_authorization_ms = (time.perf_counter() - started) * 1000
        return response

    def execute_with_authorization(self, original, authorization, sender):
        """Return denial on every invalid condition; never call sender first."""
        if not callable(sender):
            return ExecutionResult(False, "invalid_sender")
        try:
            validate_request(original)
        except (TypeError, ValueError):
            return ExecutionResult(False, "request_invalid")
        if not isinstance(authorization, dict) or authorization.get("protocol_version") != VERSION:
            return ExecutionResult(False, "authorization_invalid")
        required = {"protocol_version", "decision", "request_id", "decision_id",
                    "authorization_token", "expires_at", "policy_version",
                    "reason_codes", "risk", "evidence", "tool", "agent_id",
                    "session_id", "authorized_request"}
        if (not required <= set(authorization)
                or not isinstance(authorization["policy_version"], str)
                or not isinstance(authorization["reason_codes"], list)
                or not authorization["reason_codes"]
                or not all(isinstance(code, str) and code for code in authorization["reason_codes"])
                or not isinstance(authorization["risk"], dict)
                or not isinstance(authorization["evidence"], list)
                or authorization["tool"] != original["tool"]
                or authorization["agent_id"] != original["agent_id"]
                or authorization["session_id"] != original["session_id"]):
            return ExecutionResult(False, "authorization_invalid")
        decision = authorization.get("decision")
        if decision not in ("ALLOW", "REDACT", "REVIEW", "BLOCK"):
            return ExecutionResult(False, "authorization_invalid")
        if authorization.get("request_id") != original.get("request_id"):
            return ExecutionResult(False, "request_mismatch", decision)
        if decision in ("BLOCK", "REVIEW"):
            if (authorization["authorization_token"] is not None
                    or authorization["authorized_request"] is not None
                    or authorization["decision_id"] is not None
                    or authorization["expires_at"] is not None):
                return ExecutionResult(False, "authorization_invalid", decision)
            return ExecutionResult(False, decision.lower(), decision,
                                   review=authorization if decision == "REVIEW" else None)
        if decision == "REDACT" and not authorization.get("authorization_token"):
            return ExecutionResult(False, "redaction_not_authorized", decision)
        token = authorization.get("authorization_token")
        decision_id = authorization.get("decision_id")
        authorized = authorization.get("authorized_request")
        if (not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{20,128}", token)
                or not isinstance(decision_id, str) or not decision_id
                or not isinstance(authorized, dict)):
            return ExecutionResult(False, "authorization_invalid", decision)
        try:
            expires = datetime.fromisoformat(authorization["expires_at"])
            if expires.tzinfo is None or expires.timestamp() <= time.time():
                return ExecutionResult(False, "authorization_expired", decision)
            validate_request(authorized)
        except (KeyError, TypeError, ValueError):
            return ExecutionResult(False, "authorization_invalid", decision)
        for field in ("request_id", "agent_id", "session_id", "tool", "permissions", "timestamp"):
            if authorized[field] != original.get(field):
                return ExecutionResult(False, f"{field}_mismatch", decision)
        if decision == "ALLOW" and authorized != original:
            return ExecutionResult(False, "authorization_mismatch", decision)
        if decision == "REDACT" and (authorized["request"]["text"] == original["request"]["text"]
                                      or authorized["request"]["arguments"] != original["request"]["arguments"]):
            return ExecutionResult(False, "redaction_invalid", decision)
        consume_started = time.perf_counter()
        consumed = self._post("/v1/consume", {"authorization_token": token,
                                              "decision_id": decision_id,
                                              "authorized_request": authorized})
        if (not isinstance(consumed, dict) or consumed.get("protocol_version") != VERSION
                or consumed.get("valid") is not True
                or not isinstance(consumed.get("reason"), str)
                or consumed.get("request_id") != original["request_id"]
                or consumed.get("decision_id") != decision_id):
            return ExecutionResult(False, "authorization_not_consumed", decision)
        consume_ms = (time.perf_counter() - consume_started) * 1000
        payload = authorized["request"]
        tool_started = time.perf_counter()
        output = sender(payload["text"], payload["arguments"])
        return ExecutionResult(True, "executed", decision,
                               output=output,
                               timing_ms={"consume": consume_ms,
                                          "tool_execution": (time.perf_counter() - tool_started) * 1000})

    def invoke(self, text, arguments, sender):
        started = time.perf_counter()
        try:
            request = self.build_request(text, arguments)
        except (TypeError, ValueError):
            result = ExecutionResult(False, "invalid_request")
            self.denied_execution_count += 1
            return result
        response = self.authorize(request)
        if response is None:
            result = ExecutionResult(False, "gateway_unavailable")
        else:
            result = self.execute_with_authorization(request, response, sender)
        if not result.executed:
            self.denied_execution_count += 1
        timing = dict(result.timing_ms or {})
        timing["authorization"] = self.last_authorization_ms
        timing["gateway_total"] = (time.perf_counter() - started) * 1000
        return replace(result, timing_ms=timing)
