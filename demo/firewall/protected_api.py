"""Protected demo API for the `record.write` tool: local SQLite, no external calls.

Only the trusted adapter holds the key. Each (agent_id, operation_id) executes at
most once: the same payload returns the earlier result, a different payload is
rejected. The fault file makes the next successful write drop its HTTP reply,
so the firewall's "unknown execution" path can be shown against a real commit.
"""

import hashlib
import json
import os
import re
import secrets
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY, agent_id TEXT NOT NULL, operation_id TEXT NOT NULL,
    text TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS operations (
    agent_id TEXT NOT NULL, operation_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
    record_id INTEGER NOT NULL REFERENCES records(id),
    PRIMARY KEY (agent_id, operation_id));
"""


def connect(db_path):
    connection = sqlite3.connect(db_path, isolation_level=None, timeout=5)
    connection.executescript(SCHEMA)
    return connection


def write_record(db_path, agent_id, text, arguments):
    """Return (http_status, body). Atomic per (agent_id, operation_id)."""
    if (not isinstance(agent_id, str) or not ID.fullmatch(agent_id)
            or not isinstance(text, str) or not 0 < len(text) <= 2000
            or not isinstance(arguments, dict) or set(arguments) != {"operation_id"}
            or not isinstance(arguments["operation_id"], str)
            or not ID.fullmatch(arguments["operation_id"])):
        return 400, {"error": "invalid_record"}
    operation_id = arguments["operation_id"]
    digest = hashlib.sha256(json.dumps({"text": text, "arguments": arguments},
                                       sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    connection = connect(db_path)
    try:
        # BEGIN IMMEDIATE takes the write lock before the lookup: no check-then-insert race.
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT payload_sha256, record_id FROM operations "
            "WHERE agent_id = ? AND operation_id = ?", (agent_id, operation_id)).fetchone()
        if row is not None:
            connection.execute("COMMIT")
            if not secrets.compare_digest(row[0], digest):
                return 409, {"error": "operation_id_conflict"}
            return 200, {"status": "accepted", "record_id": row[1], "replayed": True}
        record_id = connection.execute(
            "INSERT INTO records (agent_id, operation_id, text, created_at) VALUES (?, ?, ?, ?)",
            (agent_id, operation_id, text, datetime.now(timezone.utc).isoformat())).lastrowid
        connection.execute(
            "INSERT INTO operations (agent_id, operation_id, payload_sha256, record_id) "
            "VALUES (?, ?, ?, ?)", (agent_id, operation_id, digest, record_id))
        connection.execute("COMMIT")
        return 200, {"status": "accepted", "record_id": record_id, "replayed": False}
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def dump(db_path):
    connection = connect(db_path)
    try:
        records = connection.execute(
            "SELECT id, agent_id, operation_id, text FROM records ORDER BY id").fetchall()
        operations = connection.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
    finally:
        connection.close()
    return {"records": [dict(zip(("id", "agent_id", "operation_id", "text"), row))
                        for row in records], "operations": operations}


def make_target(host, port, db_path, key, fault_path=None):
    if not isinstance(key, str) or len(key) < 16:
        raise ValueError("target key required")
    connect(db_path).close()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/records":
                return self.respond(404, {"error": "not_found"})
            if not secrets.compare_digest(self.headers.get("X-Target-Key", ""), key):
                return self.respond(403, {"error": "forbidden"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {"agent_id", "text", "arguments"}:
                    raise ValueError
            except ValueError:
                return self.respond(400, {"error": "invalid_record"})
            status, reply = write_record(db_path, body["agent_id"], body["text"],
                                         body["arguments"])
            if status == 200 and not reply["replayed"] and fault_path and os.path.exists(fault_path):
                os.remove(fault_path)
                self.close_connection = True  # committed, but the caller never hears back
                return None
            return self.respond(status, reply)

        def respond(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    db = os.getenv("TARGET_DB", "/data/records.sqlite")
    if sys.argv[1:] == ["dump"]:
        print(json.dumps(dump(db), ensure_ascii=False))
    else:
        make_target("0.0.0.0", 8767, db, os.environ["TRUSTEDGE_TARGET_KEY"],
                    str(Path(db).with_name("fault_drop_next_response"))).serve_forever()
