"""Run the checksummed firewall fixtures through real V2 (ONNX), no fake analyzer.

In-process stack: gateway (real V2 + policy + authorization) -> trusted adapter
-> SQLite record.write target from demo/firewall/protected_api.py. Every
fixture's observed result is recorded, including model failures. The REDACT
path is VERIFIED only if a PII fixture reaches the target as sanitized text.

Run: .venv/bin/python scripts/validate_firewall_real_v2.py
"""

import hashlib
import http.client
import json
import re
import socket
import sys
import tempfile
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo/firewall"))
from protected_api import dump, make_target  # noqa: E402
from trustlaya.agent_risk import PERMISSIONS  # noqa: E402
from trustlaya.api import make_server  # noqa: E402
from trustlaya.authorization import payload_hash  # noqa: E402
from trustlaya.evidence import PII_TYPES, extract  # noqa: E402
from trustlaya.guarded_tool import GuardedTool  # noqa: E402
from trustlaya.inference import Analyzer  # noqa: E402
from trustlaya.trusted_adapter import make_adapter  # noqa: E402

FIXTURES = ROOT / "data/firewall_fixtures.json"
KEY, ADAPTER_KEY, TARGET_KEY = "k" * 40, "a" * 40, "t" * 40
TOOL = {"name": "record", "operation": "write", "target": "demo://records"}
PERMS = {key: key == "database" for key in PERMISSIONS}
RULES = [{**TOOL, "permissions": PERMS, "agents": ["agent-1"]}]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post(port, path, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("POST", path, json.dumps(body).encode(),
                           {"Content-Type": "application/json", "X-TrustLaya-Key": KEY})
        return json.loads(connection.getresponse().read())
    finally:
        connection.close()


def serve(server):
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    fixtures_bytes = FIXTURES.read_bytes()
    fixtures = json.loads(fixtures_bytes)["fixtures"]
    analyzer = Analyzer("onnx", model_dir=ROOT / "models/trustlaya-s-v2",
                        onnx_path=ROOT / "models/exported/v2/trustlaya_s.onnx")
    db = str(Path(tempfile.mkdtemp()) / "records.sqlite")
    target = serve(make_target("127.0.0.1", 0, db, TARGET_KEY))
    gateway_port = free_port()
    adapter = serve(make_adapter(port=0, gateway_url=f"http://127.0.0.1:{gateway_port}",
                                 gateway_key=KEY, adapter_key=ADAPTER_KEY,
                                 target_url=f"http://127.0.0.1:{target.server_port}/records",
                                 target_key=TARGET_KEY, tool=TOOL, permissions=PERMS))
    gateway = serve(make_server(port=gateway_port, analyzer=analyzer, tool_rules=RULES, shared_key=KEY,
                                adapter_url=f"http://127.0.0.1:{adapter.server_port}",
                                adapter_key=ADAPTER_KEY, adapter_timeout=5))
    client = GuardedTool(agent_id="agent-1", session_id="fixtures", tool=TOOL, permissions=PERMS,
                         gateway_url=f"http://127.0.0.1:{gateway_port}", shared_key=KEY)

    def records(operation_id):
        return [r["text"] for r in dump(db)["records"] if r["operation_id"] == operation_id]

    results = []
    for fx in fixtures:
        out = {"id": fx["id"], "category": fx["category"], "expected": fx["expected"]}
        request = client.build_request(fx["text"], {"operation_id": fx["id"]})
        if fx["category"] in ("BENIGN", "ATTACK", "PII", "REVIEW", "MODEL_FAILURE"):
            reply = post(gateway_port, "/v1/tool", request)
            rows = records(fx["id"])
            out.update(observed=reply.get("decision"), reason_codes=reply.get("reason_codes"),
                       execution_status=reply.get("execution_status"), risk=reply.get("risk"),
                       coverage=(reply.get("coverage") or {}).get("truncated"), target_records=len(rows))
            if fx["category"] == "BENIGN":
                out["passed"] = reply.get("decision") == "ALLOW" and rows == [fx["text"]]
            elif fx["category"] == "PII":
                raw_pii = [fx["text"][e["start"]:e["end"]] for e in extract(fx["text"])
                           if e["type"] in PII_TYPES - {"PII_MENTION"}]
                out["target_text_sanitized"] = bool(rows) and all(v not in rows[0] for v in raw_pii) \
                    and "[REDACTED]" in rows[0]
                out["passed"] = reply.get("decision") == "REDACT" and out["target_text_sanitized"]
                out["original_pii_reached_target"] = any(v in t for v in raw_pii for t in rows)
            else:
                out["passed"] = reply.get("decision") == fx["expected"] and not rows
            if fx.get("known_failure"):
                out["known_failure"] = fx["known_failure"]
        elif fx["id"] == "gateway_failure_adapter_down":
            lonely = serve(make_server(port=0, analyzer=analyzer, tool_rules=RULES, shared_key=KEY,
                                       adapter_url=f"http://127.0.0.1:{free_port()}", adapter_key=ADAPTER_KEY))
            reply = post(lonely.server_port, "/v1/tool", request)
            lonely.shutdown(); lonely.server_close()
            out.update(observed=reply.get("execution_status"), decision=reply.get("decision"),
                       reason=reply.get("reason"), target_records=len(records(fx["id"])))
            out["passed"] = reply.get("execution_status") == "not_executed" and not records(fx["id"])
        elif fx["id"] == "gateway_failure_gateway_down":
            dead = GuardedTool(agent_id="agent-1", session_id="fixtures", tool=TOOL, permissions=PERMS,
                               gateway_url=f"http://127.0.0.1:{free_port()}", shared_key=KEY)
            result = dead.invoke(fx["text"], {"operation_id": fx["id"]}, lambda *_: "should not run")
            out.update(observed="NOT_EXECUTED" if not result.executed else "EXECUTED", reason=result.reason)
            out["passed"] = not result.executed and not records(fx["id"])
        else:
            authorization = client.authorize(request)
            body = {"authorization_token": authorization.get("authorization_token"),
                    "decision_id": authorization.get("decision_id"),
                    "authorized_request": authorization.get("authorized_request")}
            out["authorization_decision"] = authorization.get("decision")
            if fx["category"] == "REPLAY":
                first = client._post("/v1/consume", body) or {}
                second = client._post("/v1/consume", body) or {}
                out.update(observed=[first.get("reason"), second.get("reason")])
                out["passed"] = first.get("valid") is True and second.get("reason") == "authorization_replayed"
            else:
                changed = json.loads(json.dumps(body["authorized_request"]))
                changed["request"]["text"] = fx["mutated_text"]
                changed["payload_sha256"] = payload_hash(changed["request"])
                reply = client._post("/v1/consume", {**body, "authorized_request": changed}) or {}
                out.update(observed=reply.get("reason"))
                out["passed"] = reply.get("valid") is False and reply.get("reason") == "authorization_mismatch"
            out["target_records"] = len(records(fx["id"]))
            out["passed"] = out["passed"] and not out["target_records"]
        results.append(out)
        print(f"{'PASS' if out['passed'] else 'FAIL'}  {fx['id']}: {out.get('observed')}", flush=True)
    for server in (gateway, adapter, target):
        server.shutdown(); server.server_close()

    pii = [r for r in results if r["category"] == "PII"]
    verified = [r["id"] for r in pii if r["passed"]]
    report = {
        "model": "TrustLaya-S V2 FP32 ONNX", "versions": analyzer.versions,
        "fixtures_file": "data/firewall_fixtures.json",
        "fixtures_sha256": hashlib.sha256(fixtures_bytes).hexdigest(),
        "REAL_V2_REDACT_E2E": "VERIFIED" if verified else "NOT_VERIFIED",
        "redact_e2e_passed": f"{len(verified)}/{len(pii)}", "redact_e2e_fixtures": verified,
        "original_pii_reached_target": any(r.get("original_pii_reached_target") for r in pii),
        "passed": sum(r["passed"] for r in results), "total": len(results),
        "security_invariants_held": all(r["passed"] for r in results
                                        if r["category"] in ("ATTACK", "GATEWAY_FAILURE", "REPLAY", "PAYLOAD_MUTATION"))
        and not any(r.get("original_pii_reached_target") for r in pii),
        "results": results,
        "note": "Observed decisions are real V2 outputs; failures are kept. A fake analyzer is never used here.",
    }
    (ROOT / "reports/firewall_real_v2_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("REAL_V2_REDACT_E2E", "redact_e2e_passed", "passed", "total",
                                              "security_invariants_held")}))


if __name__ == "__main__":
    main()
