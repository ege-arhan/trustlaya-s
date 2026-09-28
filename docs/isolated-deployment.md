# Reproducible bypass test

Run `python scripts/verify_isolated_deployment.py` with Docker available.
The script creates synthetic random keys, starts four local containers and
tears them down. No production API is contacted.

```text
agent --agent_net--> gateway --target_net--> adapter --> fake target
                  authorization         target key only here
```

The agent has no `TRUSTEDGE_TARGET_KEY` or `TRUSTEDGE_ADAPTER_KEY` and no
membership in `target_net`. The target has no published port. The script
inspects the target's container IP and tries direct TCP from the agent. It
checks recorded target payloads: one harmless request and one `[REDACTED]`
replacement; the original phone number is absent. Privileged requests,
duplicate request IDs and changed payloads are denied. Stopping adapter
and gateway leaves the target count unchanged.

`demo/isolated_service.py` uses a deterministic fake analyzer. This harness
tests process and network boundaries, not model accuracy. The normal gateway
uses the real model via `scripts/serve_api.py`.

The adapter is reachable only from the gateway's private network. Its shared
key is a prototype mechanism, not a substitute for production identity or
TLS. Docker administrators can inspect synthetic keys; production secrets
need protected injection. On UNO Q, equivalent routing or firewall rules
must be verified on the physical host.
