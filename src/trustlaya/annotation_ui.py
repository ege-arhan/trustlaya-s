"""Local-only blind human review UI for the frozen V5 candidate packet."""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .v5_benchmark import ATTACK_VECTORS, BENIGN_TYPES, INTENTS, LOCATIONS

ROLES = ("A", "B", "ADMIN", "ADJUDICATOR")
LANGUAGES = {"tr", "en", "mixed", "other", "undetermined"}
FIELDS = ("intent", "attack_vector", "benign_type", "language", "attack_location",
          "contains_pii", "contains_secret", "obfuscated", "reason")
DEFAULTS = dict(intent=None, attack_vector=None, benign_type=None, language="undetermined",
                attack_location=None, contains_pii=False, contains_secret=False,
                obfuscated=False, reason="")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(password: str, salt: bytes):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 250_000).hex()


def neutral_note(text: str):
    """Observable form only; never infer an attack or benign label."""
    words = len(text.split())
    hints = []
    glossary = {
        "ignore previous instructions": "önceki yönergeleri yok say",
        "reveal the system prompt": "sistem yönergesini göster",
        "system prompt": "sistem yönergesi",
        "previous instructions": "önceki yönergeler",
        "prompt injection": "yönerge enjeksiyonu",
        "jailbreak": "güvenlik sınırını aşma",
        "security policy": "güvenlik politikası",
        "safety rules": "güvenlik kuralları",
        "tool output": "araç çıktısı",
        "user input": "kullanıcı girdisi",
        "API key": "API anahtarı",
        "private key": "özel anahtar",
        "database": "veritabanı",
        "send": "gönder",
        "summarize": "özetle",
        "explain": "açıkla",
    }
    for english, turkish in glossary.items():
        if english.casefold() in text.casefold() and len(hints) < 5:
            hints.append(f"{english} = {turkish}")
    detail = f"Yaklaşık {words} sözcük; "
    detail += "soru biçimi içeriyor" if "?" in text else "soru işareti içermiyor"
    if hints:
        detail += ". Terim yardımı: " + "; ".join(hints)
    return detail + ". Yorumunuzu yalnızca metnin bağlamına dayandırın."


def audit_value(encoded: str | None):
    """Keep label transitions, but hash optional free text rather than logging it."""
    if encoded is None:
        return None
    value = json.loads(encoded)
    reason = value.pop("reason", "")
    value["reason_sha256"] = hashlib.sha256(reason.encode()).hexdigest() if reason else None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def validate_annotation(value: dict, final: bool, *, gold=False):
    if not isinstance(value, dict) or set(value) - set(FIELDS):
        raise ValueError("Geçersiz anotasyon alanı")
    row = {**DEFAULTS, **value}
    if row["intent"] not in (INTENTS if final else INTENTS | {None}):
        raise ValueError("Birincil etiket gerekli")
    if gold and row["intent"] == "UNRESOLVED":
        raise ValueError("Belirsiz karar GOLD olamaz")
    if row["intent"] == "ATTACK":
        if row["attack_vector"] not in (ATTACK_VECTORS if final else ATTACK_VECTORS | {None}):
            raise ValueError("Saldırı türü gerekli")
        if row["benign_type"] is not None:
            raise ValueError("Uyumsuz alt etiket")
    elif row["intent"] == "BENIGN_DUAL_USE":
        if row["benign_type"] not in (BENIGN_TYPES if final else BENIGN_TYPES | {None}):
            raise ValueError("Güvenlik içeriği türü gerekli")
        if row["attack_vector"] is not None:
            raise ValueError("Uyumsuz alt etiket")
    elif row["attack_vector"] is not None or row["benign_type"] is not None:
        raise ValueError("Uyumsuz alt etiket")
    if row["language"] not in LANGUAGES:
        raise ValueError("Geçersiz dil")
    if row["attack_location"] is not None and (
            row["intent"] != "ATTACK" or row["attack_location"] not in LOCATIONS):
        raise ValueError("Geçersiz saldırı konumu")
    if any(type(row[key]) is not bool for key in ("contains_pii", "contains_secret", "obfuscated")):
        raise ValueError("Değiştiriciler doğru/yanlış olmalı")
    if not isinstance(row["reason"], str) or len(row["reason"]) > 1000:
        raise ValueError("Gerekçe çok uzun")
    return row


def alpha_nominal(pairs):
    if not pairs:
        return None
    counts = Counter(x for pair in pairs for x in pair)
    n = 2 * len(pairs)
    expected = 1 - sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))
    return None if expected == 0 else 1 - (sum(a != b for a, b in pairs) / len(pairs)) / expected


def same_labels(a, b):
    return all(a[key] == b[key] for key in FIELDS if key != "reason")


class ReviewStore:
    def __init__(self, packet: Path, manifest: Path, db: Path):
        self.packet, self.manifest, self.db = map(Path, (packet, manifest, db))
        meta = json.loads(self.manifest.read_text())
        self.checksum = sha256(self.packet)
        if self.checksum != meta["private_file_sha256"]["review_packet.jsonl"]:
            raise ValueError("Kör paketin SHA256 değeri manifestle uyuşmuyor")
        self.rows = [json.loads(line) for line in self.packet.read_text().splitlines() if line]
        if len(self.rows) != 600 or len({r["sample_id"] for r in self.rows}) != 600 or any(
                set(r) != {"sample_id", "text"} or not r["sample_id"].startswith("r-")
                for r in self.rows):
            raise ValueError("Beklenen 600 kayıtlık anonim paket bulunamadı")
        self.ids = {r["sample_id"] for r in self.rows}
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS users(role TEXT PRIMARY KEY, salt TEXT NOT NULL, hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS annotations(
                    sample_id TEXT NOT NULL, role TEXT NOT NULL, value TEXT NOT NULL,
                    submitted INTEGER NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(sample_id, role));
                CREATE TABLE IF NOT EXISTS adjudications(
                    sample_id TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    sample_id TEXT, actor TEXT NOT NULL, action TEXT NOT NULL,
                    previous_value TEXT, new_value TEXT, duration_ms INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            previous = c.execute("SELECT value FROM metadata WHERE key='packet_sha256'").fetchone()
            if previous and previous[0] != self.checksum:
                raise ValueError("Var olan veritabanı başka bir pakete ait")
            c.execute("INSERT OR IGNORE INTO metadata VALUES('packet_sha256',?)", (self.checksum,))

    def conn(self):
        c = sqlite3.connect(self.db, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def set_password(self, role, password):
        if role not in ROLES or len(password) < 12:
            raise ValueError("Rol geçersiz veya parola 12 karakterden kısa")
        with self.conn() as c:
            if c.execute("SELECT 1 FROM users WHERE role=?", (role,)).fetchone():
                raise ValueError("Bu rol zaten kurulu; hesaplar üzerine yazılmaz")
            for existing in c.execute("SELECT salt,hash FROM users"):
                if hmac.compare_digest(derive(password, bytes.fromhex(existing["salt"])), existing["hash"]):
                    raise ValueError("Her rol için farklı parola gerekli")
            salt = secrets.token_bytes(16)
            c.execute("INSERT INTO users VALUES(?,?,?)", (role, salt.hex(), derive(password, salt)))

    def login(self, role, password):
        with self.conn() as c:
            row = c.execute("SELECT salt,hash FROM users WHERE role=?", (role,)).fetchone()
        return bool(row and hmac.compare_digest(derive(password, bytes.fromhex(row["salt"])), row["hash"]))

    def count(self, role):
        with self.conn() as c:
            return c.execute("SELECT count(*) FROM annotations WHERE role=? AND submitted=1",
                             (role,)).fetchone()[0]

    def resume_index(self, role):
        if role not in ("A", "B"):
            raise ValueError("Geçersiz incelemeci")
        with self.conn() as c:
            done = {r[0] for r in c.execute(
                "SELECT sample_id FROM annotations WHERE role=? AND submitted=1", (role,))}
        return next((i for i, row in enumerate(self.rows) if row["sample_id"] not in done), 599)

    def example(self, role, index):
        if role not in ("A", "B") or not 0 <= index < len(self.rows):
            raise ValueError("Geçersiz örnek")
        row = self.rows[index]
        with self.conn() as c:
            existing = c.execute("SELECT value,submitted FROM annotations WHERE sample_id=? AND role=?",
                                 (row["sample_id"], role)).fetchone()
        return {"sample_id": row["sample_id"], "text": row["text"],
                "note": neutral_note(row["text"]), "index": index, "total": len(self.rows),
                "checksum": self.checksum, "completed": self.count(role),
                "annotation": json.loads(existing["value"]) if existing else DEFAULTS,
                "submitted": bool(existing["submitted"]) if existing else False}

    def save(self, role, sample_id, value, submitted=False, duration_ms=0):
        if role not in ("A", "B") or sample_id not in self.ids:
            raise ValueError("Geçersiz kullanıcı veya örnek")
        if type(submitted) is not bool:
            raise ValueError("Tamamlanma durumu doğru/yanlış olmalı")
        if type(duration_ms) is not int or not 0 <= duration_ms <= 86_400_000:
            raise ValueError("Geçersiz süre")
        value = validate_annotation(value, submitted)
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        with self.conn() as c:
            if c.execute("SELECT 1 FROM adjudications WHERE sample_id=?", (sample_id,)).fetchone():
                raise ValueError("Hakem kararı verilmiş örnek değiştirilemez")
            old = c.execute("SELECT value,submitted FROM annotations WHERE sample_id=? AND role=?",
                            (sample_id, role)).fetchone()
            previous = old["value"] if old else None
            if old and old["value"] == encoded and bool(old["submitted"]) == submitted:
                return {"saved": True, "submitted": submitted, "completed": self.count(role)}
            c.execute("INSERT INTO annotations VALUES(?,?,?,?,?) ON CONFLICT(sample_id,role) DO UPDATE SET "
                      "value=excluded.value, submitted=excluded.submitted, updated_at=excluded.updated_at",
                      (sample_id, role, encoded, int(submitted), stamp()))
            c.execute("INSERT INTO audit(timestamp,sample_id,actor,action,previous_value,new_value,duration_ms)"
                      " VALUES(?,?,?,?,?,?,?)",
                      (stamp(), sample_id, role, "submit" if submitted else "autosave",
                       audit_value(previous), audit_value(encoded), duration_ms))
        return {"saved": True, "submitted": submitted, "completed": self.count(role)}

    def navigation(self, role, sample_id, action, duration_ms=0):
        if role not in ("A", "B") or sample_id not in self.ids or action not in ("skip", "previous"):
            raise ValueError("Geçersiz gezinme eylemi")
        if type(duration_ms) is not int or not 0 <= duration_ms <= 86_400_000:
            raise ValueError("Geçersiz süre")
        with self.conn() as c:
            c.execute("INSERT INTO audit(timestamp,sample_id,actor,action,previous_value,new_value,duration_ms)"
                      " VALUES(?,?,?,?,?,?,?)", (stamp(), sample_id, role, action, None, None, duration_ms))
        return {"logged": True}

    def _reviews(self):
        with self.conn() as c:
            rows = c.execute("SELECT sample_id,role,value FROM annotations WHERE submitted=1").fetchall()
            adjudications = c.execute("SELECT sample_id,value FROM adjudications").fetchall()
        reviews = {(r["sample_id"], r["role"]): json.loads(r["value"]) for r in rows}
        return reviews, {r["sample_id"]: json.loads(r["value"]) for r in adjudications}

    def dashboard(self):
        reviews, resolved = self._reviews()
        pairs = []
        full_agreements = []
        matrix = {a: {b: 0 for b in sorted(INTENTS)} for a in sorted(INTENTS)}
        distribution = {"A": Counter(), "B": Counter()}
        pending, unresolved = [], []
        for row in self.rows:
            sid = row["sample_id"]
            a, b = reviews.get((sid, "A")), reviews.get((sid, "B"))
            if a:
                distribution["A"][a["intent"]] += 1
            if b:
                distribution["B"][b["intent"]] += 1
            if not (a and b):
                continue
            pairs.append((a["intent"], b["intent"]))
            full_agreements.append(same_labels(a, b))
            matrix[a["intent"]][b["intent"]] += 1
            if a["intent"] == "UNRESOLVED" or b["intent"] == "UNRESOLVED":
                unresolved.append(sid)
            if not same_labels(a, b) or a["intent"] == "UNRESOLVED":
                if sid not in resolved:
                    pending.append(sid)
        alpha = alpha_nominal(pairs)
        ready = len(pairs) == len(self.rows) and not pending and alpha is not None and alpha >= 0.80
        return {"total": len(self.rows), "a_completed": self.count("A"),
                "b_completed": self.count("B"), "both_completed": len(pairs),
                "exact_agreement": sum(full_agreements) / len(pairs) if pairs else None,
                "alpha_nominal": alpha, "distribution": {
                    k: dict(v) for k, v in distribution.items()},
                "matrix": matrix, "disagreement_count": len(pairs) - sum(full_agreements),
                "unresolved_ids": unresolved, "pending_ids": pending, "gold_ready": ready,
                "adjudicated": len(resolved)}

    def adjudication(self, sample_id):
        if sample_id not in self.ids:
            raise ValueError("Geçersiz örnek")
        reviews, resolved = self._reviews()
        a, b = reviews.get((sample_id, "A")), reviews.get((sample_id, "B"))
        if not (a and b):
            raise ValueError("İki inceleme tamamlanmadı")
        text = next(r["text"] for r in self.rows if r["sample_id"] == sample_id)
        return {"sample_id": sample_id, "text": text, "A": a, "B": b,
                "final": resolved.get(sample_id)}

    def resolve(self, sample_id, value, duration_ms=0):
        if type(duration_ms) is not int or not 0 <= duration_ms <= 86_400_000:
            raise ValueError("Geçersiz süre")
        pair = self.adjudication(sample_id)
        if same_labels(pair["A"], pair["B"]) and pair["A"]["intent"] != "UNRESOLVED":
            raise ValueError("Uyumlu karar hakemlik gerektirmez")
        row = validate_annotation(value, True, gold=True)
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self.conn() as c:
            old = c.execute("SELECT value FROM adjudications WHERE sample_id=?", (sample_id,)).fetchone()
            c.execute("INSERT INTO adjudications VALUES(?,?,?) ON CONFLICT(sample_id) DO UPDATE SET "
                      "value=excluded.value,updated_at=excluded.updated_at", (sample_id, encoded, stamp()))
            c.execute("INSERT INTO audit(timestamp,sample_id,actor,action,previous_value,new_value,duration_ms)"
                      " VALUES(?,?,?,?,?,?,?)",
                      (stamp(), sample_id, "ADJUDICATOR", "adjudicate",
                       audit_value(old["value"] if old else None), audit_value(encoded), duration_ms))
        return {"saved": True}

    def export(self, directory: Path):
        import pyarrow as pa
        import pyarrow.parquet as pq
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        reviews, resolved = self._reviews()
        annotations = [{"sample_id": sid, "annotator_id": role, **value}
                       for (sid, role), value in sorted(reviews.items())]
        def write(name, rows):
            path = directory / name
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        def parquet(name, rows):
            pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table(
                {"sample_id": pa.array([], type=pa.string()),
                 "annotator_id": pa.array([], type=pa.string())}), directory / name)
        write("annotations.json", annotations)
        parquet("annotations.parquet", annotations)
        status = self.dashboard()
        if not status["gold_ready"]:
            for name in ("gold_annotations.json", "gold_annotations.parquet",
                         "adjudication_log.json", "annotation_metrics.json"):
                (directory / name).unlink(missing_ok=True)
        if status["gold_ready"]:
            gold = []
            for item in self.rows:
                sid = item["sample_id"]
                a, b = reviews[(sid, "A")], reviews[(sid, "B")]
                chosen = resolved.get(sid, a)
                gold.append({"sample_id": sid, "review_status": "ADJUDICATED" if sid in resolved else "AGREED",
                             **{k: v for k, v in chosen.items() if k != "reason"}})
            write("gold_annotations.json", gold)
            parquet("gold_annotations.parquet", gold)
            with self.conn() as c:
                audit = [dict(r) for r in c.execute(
                    "SELECT timestamp,sample_id,actor,action,previous_value,new_value,duration_ms "
                    "FROM audit WHERE action='adjudicate' ORDER BY id")]
            write("adjudication_log.json", audit)
            write("annotation_metrics.json", status)
        paths = sorted(p for p in directory.iterdir() if p.name != "checksums.txt" and
                       p.name in {"annotations.json", "annotations.parquet", "gold_annotations.json",
                                  "gold_annotations.parquet", "adjudication_log.json", "annotation_metrics.json"})
        (directory / "checksums.txt").write_text(
            f"{self.checksum}  review_packet.jsonl\n" +
            "".join(f"{sha256(p)}  {p.name}\n" for p in paths))
        return {"files": [p.name for p in paths] + ["checksums.txt"],
                "dataset_sha256": self.checksum, "gold_ready": status["gold_ready"]}


def make_server(store: ReviewStore, html_path: Path, export_dir: Path, host="127.0.0.1", port=8766):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Anotasyon arayüzü yalnızca localhost üzerinde çalışır")
    sessions = {}
    lock = threading.Lock()
    html = Path(html_path).read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, code, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; form-action 'self'")
            self.end_headers()
            self.wfile.write(body)

        def session(self):
            cookie = self.headers.get("Cookie", "")
            token = next((p.strip().split("=", 1)[1] for p in cookie.split(";")
                          if p.strip().startswith("tl_session=")), "")
            with lock:
                info = sessions.get(token)
            return info if info and info["expires"] > time.time() else None

        def do_GET(self):
            path = urlsplit(self.path)
            if path.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; form-action 'self'")
                self.end_headers()
                self.wfile.write(html)
                return
            auth = self.session()
            if auth is None:
                self.respond(401, {"error": "Oturum açın"})
                return
            role = auth["role"]
            try:
                if path.path == "/api/me":
                    self.respond(200, {"role": role, "csrf": auth["csrf"], "checksum": store.checksum,
                                       "resume_index": store.resume_index(role) if role in ("A", "B") else None})
                elif path.path == "/api/example" and role in ("A", "B"):
                    index = int(parse_qs(path.query).get("index", ["0"])[0])
                    self.respond(200, store.example(role, index))
                elif path.path == "/api/dashboard" and role in ("ADMIN", "ADJUDICATOR"):
                    self.respond(200, store.dashboard())
                elif path.path == "/api/adjudication" and role == "ADJUDICATOR":
                    sid = parse_qs(path.query).get("sample_id", [""])[0]
                    self.respond(200, store.adjudication(sid))
                else:
                    self.respond(403, {"error": "Erişim yok"})
            except (ValueError, KeyError) as exc:
                self.respond(400, {"error": str(exc)})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Geçersiz istek boyutu")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("JSON nesnesi gerekli")
                path = urlsplit(self.path).path
                if path == "/api/login":
                    role, password = value.get("role"), value.get("password")
                    if not isinstance(role, str) or not isinstance(password, str) or not store.login(role, password):
                        self.respond(401, {"error": "Hatalı kullanıcı veya parola"})
                        return
                    existing = self.session()
                    if existing is not None and existing["role"] != role:
                        self.respond(403, {"error": "Bu tarayıcı profilinde başka rolün oturumu açık"})
                        return
                    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
                    with lock:
                        sessions[token] = {"role": role, "csrf": csrf, "expires": time.time() + 8 * 3600}
                    self.send_response(200)
                    body = json.dumps({"role": role, "csrf": csrf}).encode()
                    self.send_header("Set-Cookie", f"tl_session={token}; HttpOnly; SameSite=Strict; Path=/")
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                auth = self.session()
                if auth is None or not hmac.compare_digest(
                        self.headers.get("X-CSRF-Token", ""), auth["csrf"]):
                    self.respond(403, {"error": "Oturum veya CSRF doğrulanamadı"})
                    return
                role = auth["role"]
                if path == "/api/annotation" and role in ("A", "B"):
                    result = store.save(role, value["sample_id"], value["annotation"],
                                        value.get("submitted", False), value.get("duration_ms", 0))
                elif path == "/api/navigation" and role in ("A", "B"):
                    result = store.navigation(role, value["sample_id"], value["action"],
                                              value.get("duration_ms", 0))
                elif path == "/api/adjudication" and role == "ADJUDICATOR":
                    result = store.resolve(value["sample_id"], value["annotation"],
                                           value.get("duration_ms", 0))
                elif path == "/api/export" and role == "ADMIN":
                    result = store.export(export_dir)
                else:
                    self.respond(403, {"error": "Erişim yok"})
                    return
                self.respond(200, result)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                self.respond(400, {"error": str(exc)})
    return ThreadingHTTPServer((host, port), Handler)


def initialize_accounts(store: ReviewStore):
    print("Dört farklı kişiye ayrı parolalar atayın. Roller: A, B, ADMIN, ADJUDICATOR.")
    for role in ROLES:
        with store.conn() as c:
            if c.execute("SELECT 1 FROM users WHERE role=?", (role,)).fetchone():
                print(f"{role}: önceden kurulu, geçiliyor")
                continue
        password = getpass.getpass(f"{role} parolası (en az 12 karakter): ")
        store.set_password(role, password)
