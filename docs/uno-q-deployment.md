# Planned Arduino UNO Q deployment

## Credential-isolated mode

Set `TRUSTEDGE_MODE=unoq` for the planned Linux service. This labels the
deployment mode only; `/health` reports `device: unverified`. No physical
UNO Q execution has been performed. Run the trusted adapter in a separate
process or isolated service on the trusted side, with `TRUSTEDGE_TARGET_KEY`
present only there. The agent must not receive that key or a route to the
target. The first hardware test uses a controlled LAN gateway path, without
transparent HTTPS interception.

Required files: v2 tokenizer and calibration in `models/trustlaya-s-v2/`,
the chosen ONNX file in `models/exported/v2/`, `configs/policy.yaml`, the
trusted tool allowlist, and gateway/adapter source. Use a 64-bit Linux image
with Python 3.10+ and ONNX Runtime wheels validated on the actual board CPU;
install project dependencies in a virtual environment. Package availability
has not yet been verified on UNO Q.

Gateway environment: `TRUSTEDGE_MODE=unoq`, `TRUSTLAYA_HOST`,
`TRUSTLAYA_PORT`, `TRUSTLAYA_TOOL_RULES`, `TRUSTLAYA_AUDIT_LOG`,
`TRUSTLAYA_SHARED_KEY`, `TRUSTEDGE_ADAPTER_URL`, `TRUSTEDGE_ADAPTER_KEY`.
Start with `.venv/bin/python scripts/serve_api.py`. Adapter environment:
`TRUSTEDGE_ADAPTER_HOST`, `TRUSTEDGE_ADAPTER_PORT`, `TRUSTLAYA_GATEWAY_URL`,
`TRUSTLAYA_SHARED_KEY`, `TRUSTLAYA_TOOL_RULES`, `TRUSTEDGE_ADAPTER_KEY`,
`TRUSTEDGE_TARGET_URL`, `TRUSTEDGE_TARGET_KEY`. Start with
`.venv/bin/python scripts/serve_adapter.py`. Never put the target key in
the agent environment. Restrict access to allowlist and environment files.
Use a TLS reverse proxy for network-facing gateway traffic; the built-in
HTTP server has no TLS. Keep adapter and target on a private network that
the agent cannot reach.

Health: `python scripts/diagnose_gateway.py --url http://127.0.0.1:8765`
from a trusted console. `/health` confirms process readiness, not hardware
identity or target connectivity. Shutdown denies new operations; gateway
restart invalidates all in-memory tokens. To roll back, stop the adapter and
gateway, restore prior files and configuration, verify the agent still cannot
reach the target, then restart.

First physical test: record board revision, Linux image, Python and ONNX
Runtime versions, model hashes, idle/load CPU and memory. Run at least 100
benign and denied requests through the guarded path. Record p50/p95/p99
inference, authorization, total latency and throughput; test timeout,
gateway and adapter outage, BLOCK, REVIEW, replay and direct-route denial.
Label every saved measurement `hardware: Arduino UNO Q`; never combine it
with the Mac benchmark.

**Status:** Not installed or measured on physical UNO Q hardware. The saved
benchmarks are from a MacBook and must not be presented as board results.

Planned placement:

```text
AI agent on a separate host
  -> HTTPS front end / trusted private route
  -> UNO Q Linux: TrustLaya-S ONNX + policy + /v1/authorize + /v1/consume
  -> trusted tool adapter -> external API or MCU-controlled actuator
```

The agent must have no direct route or credentials for the protected target.
UNO Q's Linux side would run the Python gateway, tokenizer and ONNX Runtime;
its microcontroller may later guard an actuator enable. The latter is planned,
not implemented in this milestone.

## Proposed setup checklist

1. Install a supported 64-bit Linux image and Python 3.10 or newer on the
   board. Verify PyTorch, Transformers and ONNX Runtime packages are available
   for that exact ARM64 image; current package dependencies include PyTorch.
2. Copy the v2 tokenizer, `calibration.json`, `policy.yaml` and FP32 or INT8
   ONNX artifact. Verify published SHA-256 values with the manifest. Prefer
   FP32 until INT8 policy parity passes on the target.
3. Install project dependencies in a virtual environment, then set
   `TRUSTLAYA_TOOL_RULES` to a trusted allowlist JSON file,
   `TRUSTLAYA_AUDIT_LOG` to a protected local path, and optionally
   `TRUSTLAYA_AUTH_TTL` (1–60 seconds). `TRUSTLAYA_HOST` and `TRUSTLAYA_PORT`
   select the listener. `TRUSTLAYA_SHARED_KEY` (32+ random characters) is required for non-loopback
   binding. The tool adapter uses `TRUSTLAYA_GATEWAY_URL` and the same key.
4. Start `.venv/bin/python scripts/serve_api.py`. Keep the built-in server on
   loopback behind an authenticated TLS reverse proxy for network clients.
   Configure the agent's network/service permissions so the protected target
   can be reached only through the trusted adapter.
5. Run `demo/authorization_demo.py` on the board only after dependencies are
   installed, then test actual network bypass prevention and fail-closed
   behavior with the target disconnected. Never use real credentials in demos.
6. Measure model load time, batch-one p50/p95/p99 authorization latency,
   memory, CPU, thermals, power, token expiry, restart behavior and INT8
   decision parity. Record board revision and image/package versions.

The built-in HTTP server has no TLS. The in-memory single-use token store is
single-process and restart-volatile. Neither hardware feasibility nor
production security is established by the Mac tests.
