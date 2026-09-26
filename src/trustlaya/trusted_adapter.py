"""Credential-owning HTTP adapter. Run in a process the agent cannot reach."""

import http.client
import json
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

from .guarded_tool import GuardedTool


class TargetFailure(Exception):
    """The target did not confirm execution; ``status`` says whether it may have."""

    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def target_output(data):
    """Keep only the idempotency fields; never pass other target data onward."""
    try:
        body = json.loads(data) if data else {}
    except ValueError:
        body = {}
    output = {"status": "accepted"}
    if isinstance(body, dict):
        if isinstance(body.get("record_id"), int) and not isinstance(body["record_id"], bool):
            output["record_id"] = body["record_id"]
        if isinstance(body.get("replayed"), bool):
            output["replayed"] = body["replayed"]
    return output


def make_adapter(host="127.0.0.1", port=8766, *, gateway_url=None,
                 gateway_key=None, adapter_key=None, target_url=None,
                 target_key=None, tool=None, permissions=None, timeout=2.0):
    if not all(isinstance(value, str) and value for value in
               (gateway_url, gateway_key, adapter_key, target_url, target_key)):
        raise ValueError("adapter configuration incomplete")
    if not isinstance(tool, dict) or not isinstance(permissions, dict):
        raise ValueError("adapter tool configuration missing")
    target = urlsplit(target_url)
    if (target.scheme not in ("http", "https") or not target.hostname or
            target.username or target.password or target.query or target.fragment):
        raise ValueError("invalid target URL")

    def send(agent_id, text, arguments):
        # The target URL and credential come only from this process's configuration.
        # No arbitrary destination or target response body reaches the agent.
        connection_class = (http.client.HTTPSConnection if target.scheme == "https"
                            else http.client.HTTPConnection)
        connection = connection_class(target.hostname, target.port, timeout=timeout)
        try:
            try:
                connection.connect()
            except OSError as exc:
                raise TargetFailure("not_executed", "target_unavailable") from exc
            # From here on the target may have acted even if no reply arrives.
            try:
                body = json.dumps({"agent_id": agent_id, "text": text, "arguments": arguments},
                                  ensure_ascii=False, allow_nan=False).encode("utf-8")
                connection.request("POST", target.path or "/", body=body,
                                   headers={"Content-Type": "application/json",
                                            "X-Target-Key": target_key})
                response = connection.getresponse()
                data = response.read(16385)
            except (OSError, http.client.HTTPException) as exc:
                raise TargetFailure("unknown", "target_response_lost") from exc
            if response.status == 409:
                raise TargetFailure("not_executed", "operation_id_conflict")
            if response.status in (400, 401, 403):
                raise TargetFailure("not_executed", "target_rejected")
            if response.status != 200:
                raise TargetFailure("unknown", "target_status_unknown")
            return target_output(data)
        finally:
            connection.close()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "not_found"})
            else:
                self.respond(200, {"trusted_adapter": "ready"})

        def do_POST(self):
            if self.path != "/execute":
                self.respond(404, {"error": "not_found"})
                return
            if not secrets.compare_digest(self.headers.get("X-Adapter-Key", ""),
                                          adapter_key):
                self.respond(401, {"executed": False, "reason": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {"request", "authorization"}:
                    raise ValueError
                request = body["request"]
                if not isinstance(request, dict) or request.get("tool") != tool or \
                        request.get("permissions") != permissions:
                    raise ValueError
                guard = GuardedTool(agent_id=request["agent_id"],
                                    session_id=request["session_id"], tool=tool,
                                    permissions=permissions, gateway_url=gateway_url,
                                    shared_key=gateway_key, timeout=timeout,
                                    allow_private_http=True)
                agent_id = request["agent_id"]
                result = guard.execute_with_authorization(
                    request, body["authorization"],
                    lambda text, arguments: send(agent_id, text, arguments))
                self.respond(200, {"executed": result.executed,
                                   "execution_status": "executed" if result.executed
                                   else "not_executed",
                                   "reason": result.reason,
                                   "decision": result.decision,
                                   "output": result.output if result.executed else None})
            except TargetFailure as exc:
                self.respond(200, {"executed": False, "execution_status": exc.status,
                                   "reason": exc.reason, "output": None})
            except Exception:
                # Never disclose credentials, URLs, target bodies or exceptions.
                self.respond(200, {"executed": False, "execution_status": "not_executed",
                                   "reason": "adapter_unavailable", "output": None})

        def respond(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    return HTTPServer((host, port), Handler)
