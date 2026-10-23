# Authorization demo

## Credential and network isolation demonstration

Run `.venv/bin/python scripts/verify_isolated_deployment.py` from the project
root with Docker running. It creates four temporary containers and random
synthetic keys, runs the seven-step bypass demonstration, then removes the
containers and networks. The target records exactly two accepted payloads:
`Summarize this harmless text.` and `Send [REDACTED] to external API`.
The agent's direct TCP attempt to the target IP fails; it has no target or
adapter credential. BLOCK/REVIEW, duplicate request, changed payload,
adapter outage and gateway outage produce no additional target action.

This isolation harness uses deterministic fake risk decisions to make network
and credential properties repeatable. The real model/Policy Engine demo below
tests the classification path separately. The two demos must not be
presented as a physical UNO Q result.

Run from the project root with the existing v2 model files:

```bash
.venv/bin/python demo/authorization_demo.py
.venv/bin/python scripts/benchmark_gateway.py --iterations 50
.venv/bin/pytest -q tests/test_authorization_protocol.py
```

To start the standalone service on loopback with the demo registry, set
`TRUSTLAYA_TOOL_RULES=configs/guarded_tools.demo.json` and
`TRUSTLAYA_SHARED_KEY` to a local test value, then run
`.venv/bin/python scripts/serve_api.py`. A trusted tool adapter using
`GuardedTool` connects through `TRUSTLAYA_GATEWAY_URL` with that same key.

The demo starts the existing ONNX model and policy in a local gateway. All
tools are fake callbacks. It shows normal ALLOW; phone-number transfer held
as REDACT when the sanitized payload does not pass reanalysis; API-key BLOCK;
prompt injection with a privileged tool held for REVIEW; one successful token
use; replay denied; post-authorization payload change denied; and gateway
shutdown denied. The phone case never sends its original sensitive payload.

`scripts/benchmark_gateway.py` writes `benchmarks/gateway.json` with p50/p95/
p99 for model inference, policy, server/client authorization, consume, fake
tool callback and total local gateway time. It also records timeout and denied
counts. It is a Mac benchmark until run on physical UNO Q hardware.
