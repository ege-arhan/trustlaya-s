"""Start the blind annotation app. Never binds beyond loopback."""
import argparse
from pathlib import Path

from trustlaya.annotation_ui import ReviewStore, initialize_accounts, make_server

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-accounts", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    store = ReviewStore(PRIVATE / "review_packet.jsonl",
                        ROOT / "benchmarks/v5/review_packet_manifest.json",
                        PRIVATE / "human_gold_annotations.sqlite3")
    if args.init_accounts:
        initialize_accounts(store)
        return
    with store.conn() as c:
        count = c.execute("SELECT count(*) FROM users").fetchone()[0]
    if count != 4:
        raise SystemExit("Önce --init-accounts çalıştırın ve dört ayrı rolü kurun.")
    server = make_server(store, ROOT / "demo/v5_annotation_ui.html",
                         PRIVATE / "exports", port=args.port)
    print(f"Anotasyon arayüzü: http://127.0.0.1:{args.port} (yalnızca yerel)")
    server.serve_forever()


if __name__ == "__main__":
    main()
