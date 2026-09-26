"""Fail-closed client for routing a protected text action through TrustLaya-S.

The caller must prevent direct access to the protected tool or network target.
This client is a decision gate, not a transparent network firewall.
"""

import http.client
import json
import re
import secrets
from collections.abc import Callable, Mapping
from typing import TypeVar

T = TypeVar("T")


class GatewayUnavailable(RuntimeError):
    """The decision service could not provide a valid authorization."""


class GatewayDenied(RuntimeError):
    """The policy did not allow the protected action."""

    def __init__(self, action: str, reason: str):
        self.action = action
        self.reason = reason
        super().__init__(f"{action}: {reason}")


class TrustGateway:
    """Authorize exact text before calling a protected sender.

    Construct this in the trusted tool adapter. Agent-supplied metadata must
    never replace ``agent_state``. Keep the server on loopback; independently
    restrict the agent's direct access to the destination.
    """

    def __init__(self, agent_state: Mapping[str, bool], *, host: str = "127.0.0.1",
                 port: int = 8765, timeout: float = 2.0, session_id: str | None = None):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("gateway decision service must be on loopback")
        if (isinstance(port, bool) or not isinstance(port, int) or
                not 0 < port < 65536 or isinstance(timeout, bool) or
                not isinstance(timeout, (int, float)) or timeout <= 0):
            raise ValueError("invalid port or timeout")
        if not isinstance(agent_state, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, bool)
                for key, value in agent_state.items()):
            raise ValueError("agent_state must contain boolean flags")
        if session_id is not None and (not isinstance(session_id, str) or
                                       not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id)):
            raise ValueError("invalid session_id")
        self.agent_state = {**agent_state, "agent": True}
        self.host = host
        self.port = port
        self.timeout = timeout
        self.session_id = session_id or secrets.token_hex(12)

    def authorize(self, text: str) -> dict:
        if not isinstance(text, str) or not 0 < len(text) <= 2000:
            raise GatewayDenied("BLOCK", "invalid_text")
        request = {"text": text, "agent_state": self.agent_state,
                   "session_id": self.session_id}
        body = json.dumps(request, ensure_ascii=False).encode("utf-8")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        try:
            connection.request("POST", "/analyze", body=body,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read(65537)
            if response.status != 200 or len(data) > 65536:
                raise GatewayUnavailable("decision service rejected the request")
            result = json.loads(data)
        except (OSError, TimeoutError, ValueError, http.client.HTTPException) as exc:
            raise GatewayUnavailable("decision service unavailable or invalid") from exc
        finally:
            connection.close()
        if not isinstance(result, dict) or result.get("action") not in (
                "ALLOW", "REDACT", "REVIEW", "BLOCK") or not isinstance(
                    result.get("policy_rule"), str):
            raise GatewayUnavailable("decision service returned an invalid decision")
        if result["action"] != "ALLOW":
            raise GatewayDenied(result["action"], result["policy_rule"])
        return result

    def dispatch(self, text: str, sender: Callable[[str], T]) -> T:
        """Send precisely the analyzed text, once, only after ALLOW."""
        if not callable(sender):
            raise TypeError("sender must be callable")
        self.authorize(text)
        return sender(text)
