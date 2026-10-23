"""Untrusted agent side of the firewall demo. Stdlib only; sees only the gateway.

Usage (inside the agent container):
  agent.py health
  agent.py tool '{"text": "...", "operation_id": "op-1"}'
  agent.py authorize '{"text": "...", "operation_id": "op-1"}'
  agent.py post /v1/tool|/v1/consume   (JSON body on stdin)
  agent.py probe
  agent.py bench 100 "text"
"""

import hashlib
import json
import os
import socket
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

GATEWAY = "http://gateway:8765"
TOOL = {"name": "record", "operation": "write", "target": "demo://records"}
PERMISSIONS = {"shell": False, "filesystem": False, "network": False, "database": True,
               "email": False, "external_api": False, "credential_access": False}


def build(spec):
    payload = {"text": spec["text"], "arguments": {"operation_id": spec["operation_id"]}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    return {"protocol_version": "1", "request_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": "agent-1", "session_id": spec.get("session_id", "demo"),
            "tool": spec.get("tool", TOOL), "permissions": spec.get("permissions", PERMISSIONS),
            "request": payload, "payload_sha256": digest}


def post(path, body):
    """Return (reply, latency_ms); reply is None when the gateway is unreachable."""
    started = time.perf_counter()
    request = Request(GATEWAY + path, data=json.dumps(body, ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json",
                               "X-TrustLaya-Key": os.environ["TRUSTLAYA_SHARED_KEY"]})
    try:
        with urlopen(request, timeout=10) as response:
            reply = json.load(response)
    except (OSError, URLError, ValueError) as exc:
        reply = {"gateway_unreachable": True, "error": type(exc).__name__}
    return reply, (time.perf_counter() - started) * 1000


def reachable(host, port):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def probe():
    names = {}
    for name in ("target", "adapter"):
        try:
            names[name] = socket.gethostbyname(name)
        except OSError:
            names[name] = None
    return {"private_keys_in_env": sorted(k for k in os.environ
                                          if k.startswith("TRUSTEDGE_")),
            "dns": names,
            "target_by_ip": reachable(os.environ["TARGET_IP"], 8767),
            "adapter_by_ip": reachable(os.environ["ADAPTER_IP"], 8766),
            "gateway": reachable("gateway", 8765)}


def percentiles(values):
    points = statistics.quantiles(values, n=100)
    return {"p50_ms": round(points[49], 2), "p95_ms": round(points[94], 2),
            "p99_ms": round(points[98], 2), "max_ms": round(max(values), 2)}


def bench(count, text):
    timings = {"agent_total": []}
    decisions = {}
    for _ in range(count):
        reply, ms = post("/v1/tool", build({"text": text,
                                            "operation_id": f"bench-{uuid.uuid4().hex[:12]}"}))
        timings["agent_total"].append(ms)
        key = f'{reply.get("decision")}/{reply.get("execution_status")}'
        decisions[key] = decisions.get(key, 0) + 1
        for name, value in (reply.get("timing_ms") or {}).items():
            timings.setdefault(name, []).append(value)
    return {"count": count, "outcomes": decisions,
            "components": {name: {"n": len(values), **percentiles(values)}
                           for name, values in timings.items() if len(values) >= 20}}


def main():
    command = sys.argv[1]
    if command == "health":
        try:
            with urlopen(GATEWAY + "/health", timeout=2) as response:
                out = json.load(response)
        except (OSError, URLError, ValueError):
            out = {"gateway": "unreachable"}
    elif command in ("tool", "authorize"):
        request = build(json.loads(sys.argv[2]))
        reply, ms = post("/v1/" + command, request)
        out = {"request": request, "reply": reply, "latency_ms": ms}
    elif command == "post":
        reply, ms = post(sys.argv[2], json.load(sys.stdin))
        out = {"reply": reply, "latency_ms": ms}
    elif command == "probe":
        out = probe()
    elif command == "bench":
        out = bench(int(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit("unknown command")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
