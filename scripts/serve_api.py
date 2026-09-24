"""Start the local TrustLaya-S HTTP API."""

import argparse
import json
import os
from pathlib import Path

from trustlaya.api import make_server
from trustlaya.inference import Analyzer


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("TRUSTLAYA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TRUSTLAYA_PORT", "8765")))
    parser.add_argument("--model-dir", type=Path, default=root / "models/trustlaya-s-v2")
    parser.add_argument("--onnx", type=Path, default=root / "models/exported/v2/trustlaya_s.onnx")
    parser.add_argument("--audit-log", type=Path,
                        default=os.getenv("TRUSTLAYA_AUDIT_LOG"),
                        help="opt-in redacted JSONL audit path")
    parser.add_argument("--tool-rules", type=Path,
                        default=os.getenv("TRUSTLAYA_TOOL_RULES"),
                        help="trusted JSON allowlist for /v1/authorize")
    parser.add_argument("--authorization-ttl", type=int,
                        default=int(os.getenv("TRUSTLAYA_AUTH_TTL", "15")))
    args = parser.parse_args()
    backend = "onnx" if args.onnx or not args.model_dir else "torch_cpu"
    analyzer = Analyzer(backend, model_dir=args.model_dir, onnx_path=args.onnx)
    tool_rules = json.loads(args.tool_rules.read_text())["tools"] if args.tool_rules else None
    server = make_server(args.host, args.port, analyzer, args.audit_log,
                         tool_rules=tool_rules,
                         shared_key=os.getenv("TRUSTLAYA_SHARED_KEY"),
                         authorization_ttl=args.authorization_ttl,
                         adapter_url=os.getenv("TRUSTEDGE_ADAPTER_URL"),
                         adapter_key=os.getenv("TRUSTEDGE_ADAPTER_KEY"),
                         mode=os.getenv("TRUSTEDGE_MODE", "local"))
    print(f"TrustLaya-S API listening at http://{args.host}:{args.port}/analyze", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
