"""Measure local protocol latency with the real ONNX model and fake tool."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

import numpy as np

from trustlaya.api import make_server
from trustlaya.guarded_tool import GuardedTool
from trustlaya.inference import Analyzer


def percentiles(values):
    if not values:
        return None
    return {f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 95, 99)}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not 2 <= args.iterations <= 1000:
        parser.error("iterations must be 2..1000")
    rules = json.loads((root / "configs/guarded_tools.demo.json").read_text())["tools"]
    analyzer = Analyzer("onnx", model_dir=root / "models/trustlaya-s-v2",
                        onnx_path=root / "models/exported/v2/trustlaya_s.onnx")
    server = make_server(port=0, analyzer=analyzer, tool_rules=rules)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    client = GuardedTool(agent_id="demo-agent", session_id="benchmark",
                         tool={key: rules[0][key] for key in ("name", "operation", "target")},
                         permissions=rules[0]["permissions"],
                         gateway_url=f"http://127.0.0.1:{server.server_port}")
    measurements = {key: [] for key in ("model_inference", "policy", "authorization_server",
                    "authorization_client", "consume", "tool_execution", "gateway_total")}
    denied = 0
    try:
        for index in range(args.iterations + 1):
            request = client.build_request("Merhaba, toplantı yarın.", {})
            total_start = time.perf_counter()
            response = client.authorize(request)
            result = client.execute_with_authorization(request, response,
                                                       lambda _text, _args: None)
            total_ms = (time.perf_counter() - total_start) * 1000
            if not result.executed:
                denied += 1
            if index == 0:  # Warm-up is excluded from latency percentiles.
                continue
            if response is not None:
                timing = response.get("timing_ms", {})
                for key in ("model_inference", "policy"):
                    if key in timing:
                        measurements[key].append(timing[key])
                if "authorization_total" in timing:
                    measurements["authorization_server"].append(timing["authorization_total"])
            measurements["authorization_client"].append(client.last_authorization_ms)
            if result.timing_ms:
                measurements["consume"].append(result.timing_ms["consume"])
                measurements["tool_execution"].append(result.timing_ms["tool_execution"])
            measurements["gateway_total"].append(total_ms)
    finally:
        server.shutdown()
        worker.join()
        server.server_close()
    report = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "machine": platform.platform(), "architecture": platform.machine(),
              "scope": "Local Mac HTTP/JSON authorization, v2 FP32 ONNX CPU, fake no-op tool. Not UNO Q.",
              "iterations": args.iterations, "warmup_excluded": 1,
              "denied_execution_count": denied, "timeout_count": client.timeout_count,
              "latency": {key: percentiles(values) for key, values in measurements.items()}}
    destination = root / "benchmarks/gateway.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
