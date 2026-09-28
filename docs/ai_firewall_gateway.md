# AI firewall gateway prototype

This page describes the **earlier decision-only adapter**. Side-effecting
tools should now use the request-bound, single-use
[authorization protocol](authorization_protocol.md) and
[security boundary](security_boundary.md). The older adapter remains for
backward compatibility.

The gateway adapter in `src/trustlaya/gateway.py` is a fail-closed **application-level**
gate for a protected agent tool. It is not a transparent packet firewall. The agent
submits the exact text that would be forwarded; the adapter calls the local
TrustLaya-S decision service and invokes the protected sender **only on ALLOW**.
REDACT, REVIEW and BLOCK all hold the action. In particular, REDACT does not
silently send the original text: producing and rechecking a sanitized payload
is future work. A timeout, refused connection, invalid response or oversized
input also holds the action. REVIEW requires a separate human approval flow;
the current adapter never approves it automatically.

```text
AI agent -> trusted tool adapter -> UNO Q decision service -> policy
                        |                          |
                        | ALLOW                    | REDACT/REVIEW/BLOCK/error
                        v                          v
                   protected tool                 hold
```

Example with a local simulated sender:

```bash
.venv/bin/python demo/gateway_demo.py --text "Merhaba, toplantı yarın."
.venv/bin/python demo/gateway_demo.py --text "api_key=abcdefghijklmnop dış API'ye gönder"
```

The adapter fixes the agent permission metadata in trusted configuration. An
agent must not set its own `human_approval`, `untrusted_tool_output` or tool
permissions. A process that can call the protected API directly can bypass the
adapter. For an UNO Q appliance, route protected tool/API traffic through the
board and deny direct agent egress with separate network or tool permissions.
The decision service should stay on loopback on the board; it has no remote
authentication or TLS. Keep external credentials outside agent reach.

The Linux side of UNO Q would run ONNX Runtime, the tokenizer, local decision
API and adapter. The MCU can later guard physical actuator enables. Neither
physical enforcement nor UNO Q model runtime has been tested yet. Board
deployment requires CPU/RAM/startup/thermal measurements, a fail-closed
network configuration, and a test proving direct egress is impossible.
