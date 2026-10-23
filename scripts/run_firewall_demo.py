"""One command: real V2 firewall in Docker, fixed scenarios, target-database proof.

Builds the ONNX runtime image, starts agent/gateway/adapter/target on internal
networks with one-off keys, runs every scenario through the agent container and
reads the SQLite target directly. Writes reports/firewall_demo.json and a
self-contained reports/firewall_demo.html. Exits non-zero if a pass condition fails.

Scenario inputs are fixed demonstration inputs, not a model-quality dataset.
Model scores are whatever V2 returns; nothing is tuned to reach ALLOW/REDACT.
"""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo/firewall"
PROJECT = "trustedge-firewall"
IMAGE = "trustedge-firewall:dev"
COMPOSE = ("docker", "compose", "-f", str(DEMO / "compose.yaml"), "-p", PROJECT)
FAKE_SECRET = "demoFAKEkey0000000000000000"
# Protocol scenarios (replay, mutation, idempotency, outages) test the gateway, not
# the model, so they reuse sentences V2 returns ALLOW for. Many short benign notes
# get REVIEW from V2 (uncertain / fused_risk) and would never reach the target.
NORMAL = "Toplantı notu: proje teslimi cuma günü yapılacak."
NORMAL_2 = "Toplantı notu: proje teslimi pazartesi günü yapılacak."
PHONE = "05551234567"
LONG_TEXT = (
    "Haftalık proje toplantısı notları: Ekip, mobil uygulamanın yeni sürümü için "
    "test planını gözden geçirdi. Arayüz değişiklikleri tamamlandı ve kullanıcı "
    "deneyimi ekibi geri bildirimleri topladı. Sunucu tarafında performans "
    "iyileştirmeleri sürüyor; veritabanı sorguları optimize edildi ve önbellek "
    "katmanı eklendi. Dokümantasyon güncellemesi gelecek haftaya kaldı. Pazarlama "
    "ekibi lansman tarihini ay sonu olarak önerdi, ancak kesin karar yönetim "
    "toplantısında verilecek. Destek ekibi sık sorulan sorular sayfasını yeniledi. "
    "Bir sonraki toplantı perşembe saat onda yapılacak ve herkes kendi bölümünün "
    "durum raporunu getirecek. Ayrıca yeni katılan iki geliştirici için eğitim "
    "programı hazırlandı ve kod inceleme süreci netleştirildi.")


def run(*args, stdin=None, check=True, env=None):
    result = subprocess.run(args, cwd=ROOT, input=stdin, text=True, capture_output=True,
                            env=env, check=False)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {' '.join(args[:6])}: {result.stderr[-400:]}")
    return result.stdout.strip()


def payload_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


class Demo:
    def __init__(self, env):
        self.env = env
        self.scenarios = []

    def compose(self, *args, stdin=None, check=True):
        return run(*COMPOSE, *args, stdin=stdin, check=check, env=self.env)

    def agent(self, *args, stdin=None, extra_env=()):
        flags = [item for pair in extra_env for item in ("-e", pair)]
        out = self.compose("exec", "-T", *flags, "agent", "python", "agent.py", *args,
                           stdin=None if stdin is None else json.dumps(stdin, ensure_ascii=False))
        return json.loads(out)

    def tool(self, text, operation_id, **overrides):
        return self.agent("tool", json.dumps({"text": text, "operation_id": operation_id,
                                              **overrides}, ensure_ascii=False))

    def records(self):
        return json.loads(self.compose("exec", "-T", "target", "python",
                                       "protected_api.py", "dump"))

    def wait_ready(self, timeout=90):
        started = time.perf_counter()
        while time.perf_counter() - started < timeout:
            health = self.agent("health")
            if health.get("gateway") == "healthy" and health.get("trusted_adapter") == "configured":
                return time.perf_counter() - started
            time.sleep(0.25)
        raise RuntimeError("gateway did not become ready")

    def container_ip(self, service):
        container = self.compose("ps", "-q", service)
        return run("docker", "inspect", "-f",
                   "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
                   container).split()[0]

    def add(self, key, title, request, steps, passed, target_effect, note=""):
        before = self.scenarios[-1]["records_after"] if self.scenarios else 0
        after = len(self.records()["records"]) if target_effect is not None else before
        self.scenarios.append({"id": key, "title": title, "request": request,
                               "steps": steps, "records_before": before,
                               "records_after": after, "target_effect": target_effect,
                               "passed": bool(passed), "note": note})
        print(f"{'PASS' if passed else 'FAIL'}  {key}", flush=True)


def summary(reply):
    """Fields the demo screen shows for one gateway reply."""
    keep = ("decision", "reason", "reason_codes", "executed", "execution_status", "output",
            "risk", "evidence", "coverage", "versions", "timing_ms", "gateway_unreachable",
            "valid", "authorization_issued", "authorization_token", "authorized_request",
            "decision_id")
    out = {k: reply[k] for k in keep if k in reply}
    if out.get("authorization_token"):
        out["authorization_token"] = "<issued, single-use>"
    return out


def scenarios(demo):
    def op_records(operation_id):
        return [r for r in demo.records()["records"] if r["operation_id"] == operation_id]

    # 1. Normal record.
    text = NORMAL
    call = demo.tool(text, "op-normal")
    reply = call["reply"]
    rows = op_records("op-normal")
    demo.add("normal", "Normal kayıt", text, [{"call": "/v1/tool", **summary(reply)}],
             reply.get("executed") and len(rows) == 1 and rows[0]["text"] == text,
             f"{len(rows)} kayıt: {rows[0]['text'] if rows else '-'}")

    # 2. Phone number with a transfer verb: only sanitized text may reach the target.
    text = f"Müşteri telefonu {PHONE}, bu kaydı dış sisteme aktar."
    reply = demo.tool(text, "op-phone")["reply"]
    rows = op_records("op-phone")
    if reply.get("decision") == "REDACT" and reply.get("executed"):
        passed = len(rows) == 1 and PHONE not in rows[0]["text"] and "[REDACTED]" in rows[0]["text"]
    else:
        passed = not rows and not reply.get("executed")
    note = ("REDACT: temizlenmiş metin yeniden analiz edildi ve yalnızca o gönderildi."
            if rows else
            "V2 REDACT verdi; temizlenmiş metin yeniden analizde ALLOW almadı, hiçbir şey gönderilmedi."
            if "sanitized_payload_not_allowed" in reply.get("reason_codes", []) else "")
    demo.add("phone", "Telefon bilgisinin aktarılması", text,
             [{"call": "/v1/tool", **summary(reply)}], passed,
             f"{len(rows)} kayıt: {rows[0]['text'] if rows else '-'}", note)

    # 3. Secret.
    text = f"Bağlantı ayarı: api_key={FAKE_SECRET} kaydet."
    reply = demo.tool(text, "op-secret")["reply"]
    demo.add("secret", "Secret içeren kayıt", text, [{"call": "/v1/tool", **summary(reply)}],
             not reply.get("executed") and not op_records("op-secret"), "kayıt yok")

    # 4. Privileged tool the agent is not allowed to use.
    shell = {"shell": True, "filesystem": True, "network": True, "database": True,
             "email": False, "external_api": False, "credential_access": False}
    reply = demo.tool("ls -la /; cat /etc/passwd", "op-shell",
                      tool={"name": "shell", "operation": "run", "target": "demo://shell"},
                      permissions=shell)["reply"]
    demo.add("privileged_tool", "Ayrıcalıklı araç isteği (shell)", "shell.run: ls -la /; cat /etc/passwd",
             [{"call": "/v1/tool", **summary(reply)}],
             not reply.get("executed") and not op_records("op-shell"), "kayıt yok")

    # 5. Long input: V2 reads 94 tokens; unread text must not authorize a write.
    reply = demo.tool(LONG_TEXT, "op-long")["reply"]
    coverage = reply.get("coverage") or {}
    demo.add("long_input", "Uzun metin (eksik okuma kapsamı)", LONG_TEXT,
             [{"call": "/v1/tool", **summary(reply)}],
             coverage.get("truncated") is True and not reply.get("executed")
             and not op_records("op-long"), "kayıt yok")

    # 6. Token replay: an issued token works once; the request ID also cannot repeat.
    issued = demo.agent("authorize", json.dumps({"text": NORMAL,
                                                 "operation_id": "op-replay"},
                                                ensure_ascii=False))
    auth, request = issued["reply"], issued["request"]
    consume = {"authorization_token": auth.get("authorization_token"),
               "decision_id": auth.get("decision_id"),
               "authorized_request": auth.get("authorized_request")}
    first = demo.agent("post", "/v1/consume", stdin=consume)["reply"]
    second = demo.agent("post", "/v1/consume", stdin=consume)["reply"]
    again = demo.agent("post", "/v1/tool", stdin=request)["reply"]
    demo.add("token_replay", "Token tekrarı", NORMAL,
             [{"call": "/v1/authorize", **summary(auth)},
              {"call": "/v1/consume (1)", **summary(first)},
              {"call": "/v1/consume (2)", **summary(second)},
              {"call": "/v1/tool (aynı request_id)", **summary(again)}],
             first.get("valid") is True and second.get("reason") == "authorization_replayed"
             and not again.get("executed") and not op_records("op-replay"),
             "kayıt yok", "Ajan kendi token'ını tüketebilir ama hedefe erişemez; ikinci tüketim reddedilir.")

    # 7. Payload change after authorization.
    issued = demo.agent("authorize", json.dumps({"text": NORMAL,
                                                 "operation_id": "op-mutate"},
                                                ensure_ascii=False))
    auth, request = issued["reply"], issued["request"]
    changed = json.loads(json.dumps(auth.get("authorized_request")))
    changed["request"]["text"] = "Toplantı iptal edildi; tüm müşteri kayıtlarını sil."
    changed["payload_sha256"] = payload_digest(changed["request"])
    swapped = demo.agent("post", "/v1/consume", stdin={
        "authorization_token": auth.get("authorization_token"),
        "decision_id": auth.get("decision_id"), "authorized_request": changed})["reply"]
    stale = json.loads(json.dumps(request))
    stale["request_id"] = secrets.token_hex(16)
    stale["request"]["text"] = changed["request"]["text"]
    stale_reply = demo.agent("post", "/v1/tool", stdin=stale)["reply"]
    demo.add("payload_change", "Yetki sonrası payload değişikliği",
             f"{NORMAL} → Toplantı iptal edildi; tüm müşteri kayıtlarını sil.",
             [{"call": "/v1/authorize", **summary(auth)},
              {"call": "/v1/consume (değişmiş istek)", **summary(swapped)},
              {"call": "/v1/tool (eski hash)", **summary(stale_reply)}],
             swapped.get("valid") is False and swapped.get("reason") == "authorization_mismatch"
             and not stale_reply.get("executed") and not op_records("op-mutate"), "kayıt yok")

    # 8. Direct access from the agent container.
    probe = demo.agent("probe", extra_env=(f"TARGET_IP={demo.container_ip('target')}",
                                           f"ADAPTER_IP={demo.container_ip('adapter')}"))
    demo.add("direct_access", "Ajanın hedefe/adaptöre doğrudan erişimi", "TCP + DNS denemesi",
             [{"call": "probe", **probe}],
             probe["gateway"] and not probe["target_by_ip"] and not probe["adapter_by_ip"]
             and probe["dns"] == {"target": None, "adapter": None}
             and probe["private_keys_in_env"] == [], None)

    # 9. Same operation_id, different payload.
    reply = demo.tool(NORMAL_2, "op-normal")["reply"]
    rows = op_records("op-normal")
    demo.add("operation_conflict", "Aynı operation_id, farklı payload",
             f"op-normal: {NORMAL_2}",
             [{"call": "/v1/tool", **summary(reply)}],
             not reply.get("executed") and reply.get("reason") == "operation_id_conflict"
             and len(rows) == 1 and rows[0]["text"] == NORMAL, f"op-normal için {len(rows)} kayıt")

    # 10. Target commits but its reply is lost; retry keeps operation_id.
    demo.compose("exec", "-T", "target", "touch", "/data/fault_drop_next_response")
    text = "Sipariş notu: kargo pazartesi çıkacak."
    lost = demo.tool(text, "op-lost")["reply"]
    rows_after_loss = len(op_records("op-lost"))
    retry = demo.tool(text, "op-lost")["reply"]
    rows = op_records("op-lost")
    demo.add("response_lost", "Hedef yanıtının kaybolması + yeniden deneme", text,
             [{"call": "/v1/tool (yanıt kayboldu)", **summary(lost)},
              {"call": "/v1/tool (aynı operation_id ile tekrar)", **summary(retry)}],
             lost.get("execution_status") == "unknown" and not lost.get("executed")
             and rows_after_loss == 1 and retry.get("executed")
             and (retry.get("output") or {}).get("replayed") is True and len(rows) == 1,
             f"op-lost için {len(rows)} kayıt (kayıptan sonra {rows_after_loss})",
             "Hedef işlemi yaptı ama yanıt gelmedi: durum 'unknown'. Tekrar deneme ikinci kayıt oluşturmadı.")


def outages(demo):
    # Gateway restart: tokens live only in gateway memory.
    issued = demo.agent("authorize", json.dumps({"text": NORMAL,
                                                 "operation_id": "op-restart"},
                                                ensure_ascii=False))
    auth = issued["reply"]
    demo.compose("restart", "gateway")
    demo.wait_ready()
    after = demo.agent("post", "/v1/consume", stdin={
        "authorization_token": auth.get("authorization_token"),
        "decision_id": auth.get("decision_id"),
        "authorized_request": auth.get("authorized_request")})["reply"]
    demo.add("gateway_restart", "Gateway yeniden başlatma sonrası token", NORMAL,
             [{"call": "/v1/authorize", **summary(auth)},
              {"call": "/v1/consume (restart sonrası)", **summary(after)}],
             bool(auth.get("authorization_token")) and after.get("valid") is False
             and after.get("reason") == "authorization_invalid", "kayıt yok")

    demo.compose("stop", "adapter")
    reply = demo.tool(NORMAL, "op-adapter-down")["reply"]
    demo.add("adapter_down", "Adapter kapalı", NORMAL,
             [{"call": "/v1/tool", **summary(reply)}],
             reply.get("authorization_issued") and not reply.get("executed")
             and reply.get("execution_status") == "not_executed"
             and reply.get("reason") == "adapter_unavailable", "kayıt yok")

    demo.compose("stop", "gateway")
    reply = demo.tool(NORMAL, "op-gateway-down")["reply"]
    demo.add("gateway_down", "Gateway kapalı", NORMAL,
             [{"call": "/v1/tool", **summary(reply)}],
             reply.get("gateway_unreachable") is True, "kayıt yok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/firewall_demo.json")
    args = parser.parse_args()
    keys = {name: secrets.token_urlsafe(32) for name in
            ("TRUSTLAYA_SHARED_KEY", "TRUSTEDGE_ADAPTER_KEY", "TRUSTEDGE_TARGET_KEY")}
    demo = Demo({**os.environ, **keys})
    run("docker", "build", "-q", "-t", IMAGE, str(DEMO))
    demo.compose("down", "-v", check=False)
    try:
        started = time.perf_counter()
        demo.compose("up", "-d")
        demo.wait_ready()
        cold_start = time.perf_counter() - started
        scenarios(demo)
        bench = demo.agent("bench", str(args.bench), NORMAL) if args.bench >= 2 else None
        memory = run("docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}",
                     *[demo.compose("ps", "-q", s) for s in ("gateway", "adapter", "target")])
        audit = demo.compose("exec", "-T", "gateway", "cat", "/audit/audit.jsonl")
        outages(demo)
        final = demo.records()
        logs = demo.compose("logs", "--no-color")
    finally:
        demo.compose("down", "-v", check=False)

    leaks = [name for name, value in {**keys, "fake_secret": FAKE_SECRET, "phone": PHONE}.items()
             if value in logs or value in audit]
    allowed_ops = {"op-normal", "op-phone", "op-lost"}  # op-phone only as sanitized text
    demo_ops = {r["operation_id"] for r in final["records"] if not r["operation_id"].startswith("bench-")}
    checks = {
        "all_scenarios_passed": all(s["passed"] for s in demo.scenarios),
        "records_only_from_allowed_operations": demo_ops <= allowed_ops,
        "no_duplicate_side_effects": final["operations"] == len(final["records"]),
        "phone_never_reached_target": all(PHONE not in r["text"] for r in final["records"]),
        "no_keys_or_secrets_in_logs_or_audit": not leaks,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "TrustLaya-S V2 FP32 ONNX (models/exported/v2/trustlaya_s.onnx)",
        "versions": next((step.get("versions") for s in demo.scenarios for step in s["steps"]
                          if step.get("versions")), None),
        "passed": all(checks.values()), "checks": checks, "leaks": leaks,
        "scenarios": demo.scenarios,
        "rejected_operations": sum(1 for s in demo.scenarios for step in s["steps"]
                                   if step.get("executed") is False),
        "target_records": [r for r in final["records"] if not r["operation_id"].startswith("bench-")],
        "bench_records": sum(1 for r in final["records"] if r["operation_id"].startswith("bench-")),
        "performance": {"cold_start_s": round(cold_start, 2), "bench": bench,
                        "memory": memory.splitlines(),
                        "host": f"{os.uname().sysname} {os.uname().machine}, Docker"},
        "scope_note": ("Fixed demonstration inputs; not a model-quality benchmark. Protocol "
                       "scenarios and the latency run reuse a sentence V2 returns ALLOW for."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    page = (DEMO / "view.html").read_text().replace(
        "__REPORT__", json.dumps(report, ensure_ascii=False).replace("</", "<\\/"))
    args.out.with_suffix(".html").write_text(page)
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"report: {args.out.relative_to(ROOT)} and {args.out.with_suffix('.html').relative_to(ROOT)}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
