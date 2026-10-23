"""Bounded request-sequence risk tracker; stores no prompt or evidence text."""

import re

from .evidence import SECRET_TYPES


class SessionRisk:
    def __init__(self, max_steps=20):
        self.max_steps = max_steps
        self.history = []

    def preview(self, text, scores, evidence=(), metadata=None):
        metadata = metadata or {}
        lower = text.casefold()
        credential = bool(metadata.get("credential_access")) or any(
            item["type"] in SECRET_TYPES for item in evidence
        ) or bool(re.search(r"\b(?:credential|api key|secret|kimlik bilgisi|parola|şifre)\b", lower))
        external = bool(metadata.get("external_api") or metadata.get("email")) or any(
            item["type"] == "EXTERNAL_TRANSFER" for item in evidence
        ) or bool(re.search(r"\b(?:send|upload|exfiltrat\w*|gönder\w*|aktar\w*)\b", lower))
        tools = tuple(key for key in ("filesystem", "shell", "network", "database", "email", "external_api")
                      if metadata.get(key))
        prior_credential = any(step["credential"] for step in self.history)
        prior_filesystem = any("filesystem" in step["tools"] for step in self.history)
        prior_shell = any("shell" in step["tools"] for step in self.history)
        credential_chain = external and (credential or prior_credential)
        tool_chain = external and prior_filesystem and (prior_shell or metadata.get("shell"))
        risk = max(float(scores.get("security_risk", 0.0)),
                   max((step["risk"] for step in self.history), default=0.0) * 0.8)
        triggers = []
        if credential_chain:
            risk = 1.0
            triggers.append("credential_external_transfer_chain")
        if tool_chain:
            risk = max(risk, 0.9)
            triggers.append("filesystem_shell_external_chain")
        if credential and not external:
            risk = max(risk, 0.65)
        return {"score": round(min(risk, 1.0), 4), "triggers": triggers,
                "steps_seen": len(self.history) + 1, "source": "deterministic_sequence_policy",
                "_summary": {"credential": credential, "external": external,
                             "tools": tools, "risk": round(min(risk, 1.0), 4)}}

    def record(self, assessment, action):
        summary = dict(assessment["_summary"])
        summary["action"] = action
        self.history.append(summary)
        self.history = self.history[-self.max_steps:]

    def clear(self):
        self.history.clear()
