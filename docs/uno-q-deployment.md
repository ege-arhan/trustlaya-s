# Planned Arduino UNO Q deployment

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
