import hashlib
import http.client
import json
import threading
from pathlib import Path

import pytest

from trustlaya.annotation_ui import ReviewStore, make_server, neutral_note, validate_annotation


def fixture_store(tmp_path):
    packet = tmp_path / "review_packet.jsonl"
    packet.write_text("".join(json.dumps({"sample_id": f"r-{i:04d}",
                                          "text": f"Example {i}: system prompt?"}) + "\n"
                              for i in range(600)))
    digest = hashlib.sha256(packet.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"private_file_sha256": {"review_packet.jsonl": digest}}))
    store = ReviewStore(packet, manifest, tmp_path / "private/annotations.sqlite3")
    for role in ("A", "B", "ADMIN", "ADJUDICATOR"):
        store.set_password(role, f"separate-{role}-passphrase")
    return store


def review(intent="NORMAL", reason=""):
    return {"intent": intent, "attack_vector": None, "benign_type": None,
            "language": "en", "attack_location": None, "contains_pii": False,
            "contains_secret": False, "obfuscated": False, "reason": reason}


def test_packet_checksum_and_resume(tmp_path):
    store = fixture_store(tmp_path)
    assert store.example("A", 0)["checksum"] == hashlib.sha256(store.packet.read_bytes()).hexdigest()
    store.save("A", "r-0000", review("NORMAL"), submitted=True, duration_ms=140)
    reopened = ReviewStore(store.packet, store.manifest, store.db)
    assert reopened.example("A", 0)["annotation"]["intent"] == "NORMAL"
    assert reopened.example("A", 0)["submitted"]
    assert reopened.example("B", 0)["annotation"]["intent"] is None
    assert reopened.count("A") == 1
    assert reopened.resume_index("A") == 1
    assert reopened.resume_index("B") == 0
    with reopened.conn() as c:
        audit = c.execute("SELECT sample_id,actor,action,duration_ms FROM audit").fetchone()
    assert tuple(audit) == ("r-0000", "A", "submit", 140)
    store.packet.write_text(store.packet.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="SHA256"):
        ReviewStore(store.packet, store.manifest, store.db)


def test_credentials_must_be_distinct(tmp_path):
    store = fixture_store(tmp_path)
    assert store.login("A", "separate-A-passphrase")
    assert not store.login("B", "separate-A-passphrase")
    with store.conn() as c:
        c.execute("DELETE FROM users WHERE role='ADJUDICATOR'")
    with pytest.raises(ValueError, match="farklı parola"):
        store.set_password("ADJUDICATOR", "separate-A-passphrase")


def test_hierarchy_unresolved_and_gold(tmp_path):
    store = fixture_store(tmp_path)
    with pytest.raises(ValueError):
        store.save("A", "r-0000", review("ATTACK"), submitted=True)
    attack = review("ATTACK")
    attack["attack_vector"] = "DIRECT_OVERRIDE"
    store.save("A", "r-0000", attack, submitted=True)
    store.save("B", "r-0000", review("UNRESOLVED"), submitted=True)
    d = store.dashboard()
    assert d["unresolved_ids"] == ["r-0000"]
    assert d["pending_ids"] == ["r-0000"]
    with pytest.raises(ValueError, match="GOLD"):
        store.resolve("r-0000", review("UNRESOLVED"))
    store.resolve("r-0000", attack)
    with pytest.raises(ValueError, match="değiştirilemez"):
        store.save("A", "r-0000", review(), submitted=True)
    assert store.dashboard()["pending_ids"] == []
    assert not store.dashboard()["gold_ready"]  # Other 599 still await review.
    assert store.adjudication("r-0000")["A"]["intent"] == "ATTACK"


def test_reason_does_not_create_label_disagreement_and_navigation_is_audited(tmp_path):
    store = fixture_store(tmp_path)
    store.save("A", "r-0000", review(reason="Birinci gerekçe"), submitted=True)
    store.save("B", "r-0000", review(reason="İkinci gerekçe"), submitted=True)
    status = store.dashboard()
    assert status["exact_agreement"] == 1
    assert status["disagreement_count"] == 0
    assert status["pending_ids"] == []
    store.navigation("A", "r-0001", "skip", 800)
    with store.conn() as c:
        event = c.execute("SELECT action,duration_ms FROM audit ORDER BY id DESC LIMIT 1").fetchone()
    assert tuple(event) == ("skip", 800)


def test_export_and_no_raw_secret_in_audit(tmp_path):
    store = fixture_store(tmp_path)
    secret = "ghp_ThisMustNotAppearInAuditLog"
    store.save("A", "r-0000", review(reason=secret), submitted=True)
    store.save("B", "r-0000", review("UNRESOLVED"), submitted=True)
    store.resolve("r-0000", review(reason=secret))
    with store.conn() as c:
        audit = json.dumps([dict(r) for r in c.execute("SELECT * FROM audit")])
    assert secret not in audit
    result = store.export(tmp_path / "exports")
    assert not result["gold_ready"]
    assert set(result["files"]) == {"annotations.json", "annotations.parquet", "checksums.txt"}
    checks = (tmp_path / "exports/checksums.txt").read_text()
    assert store.checksum in checks
    for name in ("annotations.json", "annotations.parquet"):
        assert hashlib.sha256((tmp_path / "exports" / name).read_bytes()).hexdigest() in checks
    import pyarrow.parquet as pq
    assert pq.read_table(tmp_path / "exports/annotations.parquet").num_rows == 2


def test_completed_gold_export(tmp_path):
    store = fixture_store(tmp_path)
    for i in range(600):
        sid = f"r-{i:04d}"
        value = review() if i % 2 else review("ATTACK")
        if value["intent"] == "ATTACK":
            value["attack_vector"] = "DIRECT_OVERRIDE"
        store.save("A", sid, value, submitted=True)
        store.save("B", sid, value, submitted=True)
    assert store.dashboard()["gold_ready"]
    result = store.export(tmp_path / "exports")
    assert result["gold_ready"]
    gold = json.loads((tmp_path / "exports/gold_annotations.json").read_text())
    assert len(gold) == 600 and all(r["intent"] != "UNRESOLVED" for r in gold)
    import pyarrow.parquet as pq
    assert pq.read_table(tmp_path / "exports/gold_annotations.parquet").num_rows == 600
    assert (tmp_path / "exports/annotation_metrics.json").exists()


def test_http_role_isolation_and_csrf(tmp_path):
    store = fixture_store(tmp_path)
    page = Path(__file__).resolve().parents[1] / "demo/v5_annotation_ui.html"
    server = make_server(store, page, tmp_path / "exports", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/")
        page_response = conn.getresponse()
        assert page_response.status == 200
        page_body = page_response.read().decode()
        assert "Kör, yerel inceleme" in page_body
        assert "TrustLaya-S" not in page_body
        def request(method, path, value=None, headers=None):
            conn.request(method, path, json.dumps(value) if value is not None else None,
                         {"Content-Type": "application/json", **(headers or {})})
            resp = conn.getresponse()
            body = json.loads(resp.read())
            return resp, body
        resp, login = request("POST", "/api/login",
                              {"role": "A", "password": "separate-A-passphrase"})
        cookie = resp.getheader("Set-Cookie").split(";", 1)[0]
        resp, body = request("GET", "/api/example?index=0", headers={"Cookie": cookie})
        assert resp.status == 200 and body["sample_id"] == "r-0000"
        assert not any(key in body for key in ("source", "model", "confidence", "B"))
        resp, _ = request("GET", "/api/dashboard", headers={"Cookie": cookie})
        assert resp.status == 403
        conn.request("POST", "/api/login",
                     json.dumps({"role": "B", "password": "separate-B-passphrase"}),
                     {"Content-Type": "application/json", "Cookie": cookie})
        switched = conn.getresponse()
        assert switched.status == 403
        switched.read()
        resp, _ = request("POST", "/api/annotation", {"sample_id": "r-0000",
                            "annotation": review(), "submitted": True}, {"Cookie": cookie})
        assert resp.status == 403
        resp, _ = request("POST", "/api/annotation", {"sample_id": "r-0000",
                    "annotation": review(), "submitted": True},
                    {"Cookie": cookie, "X-CSRF-Token": login["csrf"]})
        assert resp.status == 200
        assert store.count("A") == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_keyboard_shortcuts_visible_and_neutral_note():
    html = (Path(__file__).resolve().parents[1] / "demo/v5_annotation_ui.html").read_text()
    for key in ('"1"', '"2"', '"3"', '"a"', '"s"', '"d"', '"p"', '"g"', '"o"', '" "', "ArrowLeft", "Enter"):
        assert key in html
    assert "Kısayollar" in html and "Bu ne demek?" in html
    note = neutral_note("What is a system prompt?")
    assert "sistem yönergesi" in note
    assert "saldırı" not in note.casefold()
    with pytest.raises(ValueError):
        validate_annotation({"intent": "UNKNOWN"}, True)
