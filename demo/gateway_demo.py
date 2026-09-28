"""Local AI firewall demo; the protected action is a print, not a network call."""

import argparse
from pathlib import Path
from threading import Thread

from trustlaya.api import make_server
from trustlaya.gateway import GatewayDenied, GatewayUnavailable, TrustGateway
from trustlaya.inference import Analyzer


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--untrusted-tool-output", action="store_true")
    args = parser.parse_args()
    analyzer = Analyzer("onnx", model_dir=root / "models/trustlaya-s-v2",
                        onnx_path=root / "models/exported/v2/trustlaya_s.onnx")
    server = make_server(port=0, analyzer=analyzer)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        gateway = TrustGateway({"external_api": True,
                                "untrusted_tool_output": args.untrusted_tool_output},
                               port=server.server_port)
        try:
            gateway.dispatch(args.text, lambda text: print(f"FORWARDED: {text}"))
        except GatewayDenied as exc:
            print(f"HELD: {exc.action} ({exc.reason})")
        except GatewayUnavailable:
            print("HELD: decision service unavailable")
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


if __name__ == "__main__":
    main()
