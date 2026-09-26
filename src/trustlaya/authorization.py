"""Version 1 request binding and single-use, short-lived opaque authorizations."""

import hashlib
import json
import re
import secrets
import threading
import time
from datetime import datetime

from .agent_risk import PERMISSIONS
from .evidence import PII_TYPES

VERSION = "1"
MAX_CLOCK_SKEW_SECONDS = 30
ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
TOOL_FIELD = re.compile(r"[^\x00-\x1f\x7f]{1,256}\Z")


class ProtocolError(ValueError):
    pass


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("invalid_json_value") from exc


def payload_hash(payload):
    return hashlib.sha256(canonical(payload)).hexdigest()


def validate_request(request, now=None):
    if not isinstance(request, dict) or set(request) != {
            "protocol_version", "request_id", "timestamp", "agent_id", "session_id",
            "tool", "payload_sha256", "permissions", "request"}:
        raise ProtocolError("invalid_request_schema")
    if request["protocol_version"] != VERSION:
        raise ProtocolError("unsupported_version")
    for key in ("request_id", "agent_id", "session_id"):
        if not isinstance(request[key], str) or not ID.fullmatch(request[key]):
            raise ProtocolError(f"invalid_{key}")
    if not isinstance(request["timestamp"], str):
        raise ProtocolError("invalid_timestamp")
    try:
        instant = datetime.fromisoformat(request["timestamp"])
        if instant.tzinfo is None or abs(instant.timestamp() - (
                time.time() if now is None else now)) > MAX_CLOCK_SKEW_SECONDS:
            raise ValueError
    except ValueError as exc:
        raise ProtocolError("stale_or_invalid_timestamp") from exc
    tool = request["tool"]
    if not isinstance(tool, dict) or set(tool) != {"name", "operation", "target"}:
        raise ProtocolError("invalid_tool")
    if any(not isinstance(tool[k], str) or not TOOL_FIELD.fullmatch(tool[k])
           for k in ("name", "operation", "target")):
        raise ProtocolError("invalid_tool")
    permissions = request["permissions"]
    if not isinstance(permissions, dict) or set(permissions) != set(PERMISSIONS) or any(
            not isinstance(value, bool) for value in permissions.values()):
        raise ProtocolError("invalid_permissions")
    payload = request["request"]
    if not isinstance(payload, dict) or set(payload) != {"text", "arguments"} or not isinstance(
            payload["text"], str) or not 0 < len(payload["text"]) <= 2000 or not isinstance(
                payload["arguments"], dict):
        raise ProtocolError("invalid_payload")
    operation_id = payload["arguments"].get("operation_id")
    if "operation_id" in payload["arguments"] and (
            not isinstance(operation_id, str) or not ID.fullmatch(operation_id)):
        raise ProtocolError("invalid_operation_id")
    if len(canonical(payload)) > 8192:
        raise ProtocolError("payload_too_large")
    digest = request["payload_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or not secrets.compare_digest(
            digest, payload_hash(payload)):
        raise ProtocolError("payload_hash_mismatch")
    return request


def sanitize_text(text, evidence):
    """Remove concrete PII spans; a mere PII mention cannot be sanitized."""
    spans = sorted((int(item["start"]), int(item["end"])) for item in evidence
                   if item.get("type") in PII_TYPES - {"PII_MENTION"}
                   and 0 <= item.get("start", -1) < item.get("end", -1) <= len(text))
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    for start, end in reversed(merged):
        text = text[:start] + "[REDACTED]" + text[end:]
    return text


class AuthorizationStore:
    """In-memory authority. Restart invalidates all tokens; no shared-state claims."""

    def __init__(self, ttl_seconds=15, clock=None, monotonic_clock=None):
        if not 1 <= ttl_seconds <= 60:
            raise ValueError("ttl_seconds must be 1..60")
        self.ttl_seconds = ttl_seconds
        self.clock = clock or time.time
        self.monotonic_clock = monotonic_clock or time.monotonic
        self.lock = threading.Lock()
        self.records = {}
        self.request_ids = {}

    def _prune(self, now):
        self.records = {k: v for k, v in self.records.items() if v["expires_mono"] >= now}
        self.request_ids = {k: v for k, v in self.request_ids.items() if v >= now}

    def reserve(self, request_id):
        with self.lock:
            now = self.monotonic_clock()
            self._prune(now)
            if request_id in self.request_ids or len(self.request_ids) >= 4096:
                raise ProtocolError("duplicate_or_capacity")
            self.request_ids[request_id] = now + self.ttl_seconds + MAX_CLOCK_SKEW_SECONDS

    def issue(self, request):
        with self.lock:
            now = self.clock()
            mono = self.monotonic_clock()
            token = secrets.token_urlsafe(32)
            token_digest = hashlib.sha256(token.encode()).hexdigest()
            decision_id = secrets.token_hex(16)
            expires = now + self.ttl_seconds
            self.records[token_digest] = {
                "request_digest": payload_hash(request), "request_id": request["request_id"],
                "decision_id": decision_id, "expires_mono": mono + self.ttl_seconds,
                "consumed": False,
            }
            return token, decision_id, expires

    def consume(self, token, request, decision_id):
        if not isinstance(token, str) or not token or len(token) > 128:
            return False, "authorization_invalid"
        if not isinstance(decision_id, str):
            return False, "authorization_invalid"
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            record = self.records.get(digest)
            if record is None:
                return False, "authorization_invalid"
            if record["consumed"]:
                return False, "authorization_replayed"
            if self.monotonic_clock() > record["expires_mono"]:
                return False, "authorization_expired"
            if not secrets.compare_digest(record["decision_id"], decision_id):
                return False, "authorization_invalid"
            if not secrets.compare_digest(record["request_digest"], payload_hash(request)):
                return False, "authorization_mismatch"
            record["consumed"] = True
            return True, "authorization_consumed"
