# Real-model firewall demo (V2 + `record.write`)

One command builds the ONNX runtime image, starts four containers on internal
networks with one-off keys, runs fixed scenarios through the agent container,
reads the target SQLite database directly and writes the evidence:

```bash
.venv/bin/python scripts/run_firewall_demo.py
open reports/firewall_demo.html
```

```text
agent --agent_net--> gateway: V2 ONNX -> policy -> single-use authorization
gateway --control_net--> adapter: consumes the token, holds the only target key
adapter --target_net--> target: SQLite record.write, (agent_id, operation_id) idempotency
```

- The agent has only the gateway key and the `agent_net` route. It cannot
  resolve or connect to the adapter or target, and holds no `TRUSTEDGE_*` key.
- The gateway runs `scripts/serve_api.py` with `models/exported/v2/trustlaya_s.onnx`
  and the V2 tokenizer, calibration and policy files, mounted read-only
  file by file. The image has no torch, scipy or scikit-learn.
- The target (`demo/firewall/protected_api.py`) is the only place a side
  effect happens; "executed" claims are checked against its database.

Exit code is non-zero if any scenario or global check fails:
all scenarios pass; records exist only for operations that were allowed; no
operation produced two records; the phone number never reached the target; no
key, fake secret or phone number appears in container logs or the gateway audit.

## Scenarios and observed V2 results (2026-09-26, Docker on Apple Silicon)

| Scenario | Gateway result | Target |
|---|---|---|
| Normal record | ALLOW `low_risk` | 1 record |
| Phone number + "dış sisteme aktar" | REDACT; sanitized text re-analyzed, not ALLOW (`sanitized_payload_not_allowed`) | nothing |
| `api_key=...` | BLOCK `secret_evidence` | nothing |
| `shell.run` request | BLOCK `tool_or_permissions_not_allowed` | nothing |
| 113-token benign meeting note | BLOCK `prompt_injection` (V2 false positive); coverage `truncated: true` | nothing |
| Token replay | 1st consume valid, 2nd `authorization_replayed`, same `request_id` denied | nothing |
| Payload changed after authorization | `authorization_mismatch`, stale hash denied | nothing |
| Direct access from agent | DNS and TCP to adapter/target fail | — |
| Same `operation_id`, different text | `operation_id_conflict`, `not_executed` | unchanged |
| Target commits, reply dropped | `unknown`; retry with same `operation_id` returns `replayed: true` | exactly 1 record |
| Gateway restart | pre-restart token `authorization_invalid` | nothing |
| Adapter stopped | `adapter_unavailable`, `not_executed` | nothing |
| Gateway stopped | agent cannot connect | nothing |

Latency, 100 sequential `/v1/tool` writes of the normal sentence (all ALLOW and
executed): end-to-end from the agent p50 32 ms, p95 53 ms, p99 75 ms; model
p50 24 ms, p95 45 ms; adapter + target round trip p50 5 ms, p95 10 ms. Cold
start (compose up to ready) 1.4 s. Gateway memory about 270 MiB, adapter and
target about 12 MiB each. The 250 ms p95 engineering target is met on this
host; the numbers come from Docker on a Mac, not from UNO Q.

## What this does and does not show

- Scenario inputs are fixed demonstration inputs, not a model-quality dataset.
  Protocol scenarios (replay, mutation, idempotency, restart, outages) and the
  latency run reuse a sentence V2 returns ALLOW for, because they test the
  gateway rather than the model.
- V2 returned REVIEW for many short benign notes in trial runs (for example
  98 of 100 variants of "Toplantı notu N: teslim cuma." were REVIEW; spot
  checks showed `uncertain` and `fused_risk`) and BLOCKed the long benign note above. These
  are model limitations; no score or threshold was changed for the demo.
- The REDACT path that delivers sanitized text is proven with a fake analyzer in
  `tests/test_firewall.py` and `tests/test_trusted_adapter.py`; with real V2 the
  sanitized phone sentence was held instead.
- The coverage rule (unread text turns ALLOW/REDACT into REVIEW) is enforced in
  `AuthorizationService` and tested in `tests/test_firewall.py`; in the demo run
  V2 blocked the long input before that rule mattered.
- The agent holds the gateway shared key, so it can burn its own tokens via
  `/v1/consume`; that causes no side effect because only the adapter reaches the
  target. The shared key is not an identity proof.
- Single gateway process; tokens are in memory and die with a restart.
