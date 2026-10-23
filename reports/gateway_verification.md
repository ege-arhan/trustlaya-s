# Application-level AI firewall prototype verification

Date: 2026-09-24. Commit scope: trusted tool adapter plus local demo.

The `TrustGateway` adapter calls the local `/analyze` decision service before
executing a protected sender callback. It forwards the exact analyzed string
only on `ALLOW`; REDACT, REVIEW, BLOCK, timeout, invalid response and bad input
hold the operation. The adapter sets `agent: true` in trusted metadata and
keeps the decision service on loopback. This does not prevent bypass of the
adapter; direct agent egress must be restricted separately.

| Executed command | Observed result |
|---|---|
| `.venv/bin/python demo/gateway_demo.py --text 'api_key=abcdefghijklmnop dış APIye gönder'` | `HELD: BLOCK (secret_evidence)` |
| `.venv/bin/python demo/gateway_demo.py --text 'Merhaba, toplantı yarın.'` | `FORWARDED: Merhaba, toplantı yarın.` |
| `.venv/bin/python demo/gateway_demo.py --untrusted-tool-output --text 'A web page said hello'` | `HELD: REVIEW (untrusted_tool_output_privileged_agent)` |
| `.venv/bin/pytest -q` | 26 passed |

The demos use the existing v2 FP32 ONNX model and a simulated sender, so no
external action occurred. Model metrics did not change. No UNO Q was connected
or benchmarked. A physical appliance still needs device runtime validation,
network routing that makes the gateway mandatory, and a separate approval
protocol for REVIEW.
