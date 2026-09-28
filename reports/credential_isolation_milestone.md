# Credential isolation milestone — 2026-09-25

TrustLaya-S was not retrained. The existing model, policy and authorization
protocol remain the decision path. `/v1/tool` now forwards authorized requests
to a separate trusted adapter process. Only the adapter has the target API
credential and fixed target URL. The adapter consumes the single-use gateway
authorization before issuing an authenticated request and never returns the
target's raw response body to the agent.

The reproducible Docker harness uses a deterministic fake analyzer so it
measures deployment isolation rather than ML quality. Two internal Docker
networks place the agent with the gateway and the adapter with the target.
The gateway bridges the networks. The agent has neither adapter nor target
key, and no target network membership. On this Mac's Docker runtime, direct
TCP from the agent to the target's inspected IP failed. Normal and sanitized
PII requests reached the fake target. The target recorded exactly two texts:
`Summarize this harmless text.` and `Send [REDACTED] to external API`.
Privileged action, duplicate request ID, payload mutation, adapter outage and
gateway outage caused no additional target action. Target count remained 2.

Local tests: 68/68 passed, including eight new trusted-adapter tests.
`scripts/verify_isolated_deployment.py` passed. These are local Mac/Docker
results, not UNO Q hardware measurements. The earlier Mac latency benchmark
does not include this new adapter hop; no new latency claim is made.

Limitations: the Docker harness uses fake risk decisions; the existing
`demo/authorization_demo.py` separately exercises the real v2 ONNX model and
policy. The built-in HTTP server has no TLS, shared keys are prototype-grade,
and Docker/host administrators can inspect container environments. A
different deployment must reproduce the same route and credential separation.
No real UNO Q hardware was connected or measured. Target-side idempotency
remains necessary for exactly-once effects after uncertain network outcomes.
