"""TrustLaya model/policy integration for request-bound tool authorizations."""

import hashlib
import json
import time
from datetime import datetime, timezone

from .agent_risk import PERMISSIONS
from .authorization import (VERSION, AuthorizationStore, ProtocolError,
                            payload_hash, sanitize_text, validate_request)


def _utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def coverage_complete(result):
    """True only when the analyzer reports that it read every token."""
    coverage = result.get("coverage")
    return (isinstance(coverage, dict) and coverage.get("truncated") is False
            and isinstance(coverage.get("total_tokens"), int)
            and coverage.get("read_tokens") == coverage["total_tokens"])


def validate_tool_rules(rules):
    if not isinstance(rules, list):
        raise ValueError("tool_rules must be a list")
    checked = {}
    for rule in rules:
        if not isinstance(rule, dict) or not {"name", "operation", "target", "permissions", "agents"} <= set(rule) or set(rule) - {"name", "operation", "target", "permissions", "agents", "untrusted_tool_output"}:
            raise ValueError("invalid tool rule")
        if (not all(isinstance(rule[k], str) and rule[k] for k in ("name", "operation", "target"))
                or not isinstance(rule["permissions"], dict)
                or set(rule["permissions"]) != set(PERMISSIONS)
                or any(not isinstance(v, bool) for v in rule["permissions"].values())
                or not isinstance(rule["agents"], list)
                or not rule["agents"]
                or any(not isinstance(a, str) or not a for a in rule["agents"])
                or not isinstance(rule.get("untrusted_tool_output", False), bool)):
            raise ValueError("invalid tool rule")
        key = (rule["name"], rule["operation"], rule["target"])
        if key in checked:
            raise ValueError("duplicate tool rule")
        checked[key] = rule
    return checked


class AuthorizationService:
    def __init__(self, analyzer, tool_rules, *, ttl_seconds=15, clock=None,
                 monotonic_clock=None):
        self.analyzer = analyzer
        self.rules = validate_tool_rules(tool_rules)
        self.store = AuthorizationStore(ttl_seconds, clock, monotonic_clock)
        policy = getattr(analyzer, "policy", {})
        self.policy_version = hashlib.sha256(
            json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]

    def _audit(self, request, *, decision, reasons, risk=None, issued=False,
               attempted=False, allowed=False, denial=None, consumed=False):
        tool = request["tool"]
        return {"timestamp": _utc(self.store.clock()),
                "request_id": request["request_id"], "agent_id": request["agent_id"],
                "session_id": request["session_id"], "tool": tool["name"],
                "operation": tool["operation"], "target": tool["target"],
                "decision": decision, "policy_version": self.policy_version,
                "risk_summary": risk or {}, "reason_codes": reasons,
                "authorization_issued": issued, "execution_attempted": attempted,
                "execution_allowed": allowed, "denial_reason": denial,
                "token_consumed": consumed}

    def _boundary_denial(self, request, reason, trusted_tool=False):
        response = {"protocol_version": VERSION, "decision": "BLOCK",
                    "request_id": request["request_id"], "decision_id": None,
                    "authorization_token": None, "expires_at": None,
                    "policy_version": self.policy_version, "reason_codes": [reason],
                    "risk": {}, "evidence": [], "tool": request["tool"],
                    "agent_id": request["agent_id"],
                    "session_id": request["session_id"], "authorized_request": None}
        audit_request = request if trusted_tool else {
            **request, "tool": {"name": "<untrusted>", "operation": "<untrusted>",
                                "target": "<untrusted>"}}
        return response, self._audit(audit_request, decision="BLOCK",
                                     reasons=[reason], denial=reason)

    def authorize(self, request, session=None):
        started = time.perf_counter()
        validate_request(request, self.store.clock())
        tool = request["tool"]
        rule = self.rules.get((tool["name"], tool["operation"], tool["target"]))
        if (rule is None or request["agent_id"] not in rule["agents"] or
                request["permissions"] != rule["permissions"]):
            return self._boundary_denial(request, "tool_or_permissions_not_allowed")
        try:
            self.store.reserve(request["request_id"])
        except ProtocolError as exc:
            return self._boundary_denial(request, str(exc), trusted_tool=True)
        metadata = {**rule["permissions"], "agent": True, "human_approval": False,
                    "untrusted_tool_output": rule.get("untrusted_tool_output", False)}
        payload = request["request"]
        # operation_id is an idempotency key for the target, not content to analyze.
        kept = {k: v for k, v in payload["arguments"].items() if k == "operation_id"}
        arguments = {k: v for k, v in payload["arguments"].items() if k != "operation_id"}
        analysis_text = payload["text"]
        if arguments:
            analysis_text += "\n" + json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        if len(analysis_text) > 2000:
            raise ProtocolError("analysis_text_too_long")
        result = self.analyzer.analyze(analysis_text, metadata, session=session)
        decision = result["action"]
        if decision not in ("ALLOW", "REDACT", "REVIEW", "BLOCK"):
            raise ProtocolError("invalid_policy_decision")
        reasons = [result["policy_reason"]]
        if decision in ("ALLOW", "REDACT") and not coverage_complete(result):
            # Unread text was never judged; a protected action must not run on it.
            decision = "REVIEW"
            reasons.append("incomplete_analysis_coverage")
        effective = request
        if decision == "REDACT":
            sanitized = sanitize_text(payload["text"], result["evidence"])
            if sanitized == payload["text"] or arguments:
                reasons.append("redaction_not_safe")
            else:
                safe_result = self.analyzer.analyze(sanitized, metadata)
                if isinstance(result.get("timing_ms"), dict) and isinstance(
                        safe_result.get("timing_ms"), dict):
                    for name in ("model_inference", "policy", "analysis_total"):
                        result["timing_ms"][name] += safe_result["timing_ms"].get(name, 0)
                if safe_result["action"] == "ALLOW" and coverage_complete(safe_result):
                    safe_payload = {"text": sanitized, "arguments": kept}
                    effective = {**request, "request": safe_payload,
                                 "payload_sha256": payload_hash(safe_payload)}
                else:
                    reasons.append("sanitized_payload_not_allowed")
        issued = decision == "ALLOW" or (decision == "REDACT" and effective is not request)
        response = {"protocol_version": VERSION, "decision": decision,
                    "request_id": request["request_id"], "decision_id": None,
                    "authorization_token": None, "expires_at": None,
                    "policy_version": self.policy_version, "reason_codes": reasons,
                    "risk": {k: round(float(result[k]), 4) for k in (
                        "pii", "secret", "prompt_injection", "dangerous_instruction",
                        "privacy_risk", "security_risk", "ethics_risk", "oversight_risk",
                        "data_governance_risk")},
                    "evidence": [{"type": e["type"], "start": e["start"], "end": e["end"]}
                                 for e in result["evidence"]],
                    "tool": tool, "agent_id": request["agent_id"],
                    "session_id": request["session_id"],
                    "authorized_request": None,
                    "coverage": result.get("coverage"), "versions": result.get("versions")}
        if isinstance(result.get("timing_ms"), dict):
            response["timing_ms"] = dict(result["timing_ms"])
        if issued:
            token, decision_id, expires = self.store.issue(effective)
            response.update(authorization_token=token, decision_id=decision_id,
                            expires_at=_utc(expires), authorized_request=effective)
        response.setdefault("timing_ms", {})["authorization_total"] = (
            time.perf_counter() - started) * 1000
        return response, self._audit(request, decision=decision, reasons=reasons,
                                     risk=response["risk"], issued=issued,
                                     denial=None if issued else decision.lower())

    def consume(self, token, request, decision_id):
        request_valid = False
        try:
            validate_request(request, self.store.clock())
            request_valid = True
            valid, reason = self.store.consume(token, request, decision_id)
        except ProtocolError as exc:
            valid, reason = False, str(exc)
        response = {"protocol_version": VERSION, "valid": valid,
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "decision_id": decision_id if isinstance(decision_id, str) else None,
                    "reason": reason}
        event = None
        if request_valid:
            tool = request["tool"]
            rule = self.rules.get((tool["name"], tool["operation"], tool["target"]))
            trusted = (rule is not None and request["agent_id"] in rule["agents"]
                       and request["permissions"] == rule["permissions"])
            audit_request = request if trusted else {
                **request, "tool": {"name": "<untrusted>", "operation": "<untrusted>",
                                    "target": "<untrusted>"}}
            event = self._audit(audit_request, decision="ALLOW" if valid else "DENY",
                                reasons=[reason], attempted=True, allowed=valid,
                                denial=None if valid else reason, consumed=valid)
        return response, event
