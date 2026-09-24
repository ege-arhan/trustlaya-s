"""Start the local TrustLaya-S HTTP API."""

import argparse
from pathlib import Path

from trustlaya.api import make_server
from trustlaya.inference import Analyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--audit-log", type=Path, help="opt-in redacted JSONL audit path")
    args = parser.parse_args()
    backend = "onnx" if args.onnx or not args.model_dir else "torch_cpu"
    analyzer = Analyzer(backend, model_dir=args.model_dir, onnx_path=args.onnx)
    server = make_server(args.host, args.port, analyzer, args.audit_log)
    print(f"TrustLaya-S API listening at http://{args.host}:{args.port}/analyze", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
