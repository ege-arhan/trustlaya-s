# Security boundary

## Credential-isolated execution (prototype)

```text
Agent (no target key, agent network only)
  -> gateway /v1/tool (model -> existing policy -> authorization)
  -> trusted adapter (consumes authorization, owns target key)
  -> protected target (target network only)
```

The model supplies risk signals; the existing policy chooses ALLOW, REDACT,
REVIEW or BLOCK; authorization binds a short-lived single-use decision to the
exact request; the adapter alone makes authenticated target calls. Its target
URL is fixed by trusted configuration, and it never forwards target response
bodies. The gateway returns only a non-sensitive success marker.

The Docker harness places the agent only on `agent_net`, the adapter and
target only on `target_net`, and the gateway on both. Neither target nor
adapter has a published port. A direct connection from the agent to the
target's inspected IP was denied in the local Docker test. These network
rules are a **deployment property**, not a Python library guarantee. The
harness uses a deterministic fake analyzer to test network isolation; the
normal server still loads the real model.

Keep `TRUSTEDGE_TARGET_KEY` only in the adapter environment, and
`TRUSTEDGE_ADAPTER_KEY` only in gateway and adapter environments. The agent
receives neither. A Docker or host administrator can inspect environments;
this prototype does not defend against that administrator. Sharing a process,
publishing the adapter port, or giving the agent a target route would break
the intended isolation.

If model, policy, authorization, gateway, adapter or target is unavailable,
the controlled path denies execution. A target action may still happen before
its response is lost; target-side idempotency is required to prevent a new
request ID from repeating that action.

## Earlier library-level guard

The existing `GuardedTool` remains available for callers that isolate its
sender callback themselves. Its flow is:

```text
Caller with trusted sender callback
  -> trusted tool adapter / GuardedTool
  -> POST /v1/authorize on TrustLaya gateway
  -> existing model -> existing policy -> authorization store
  -> POST /v1/consume (single use)
  -> protected tool callback
```

**Four distinct responsibilities:** The model produces risk signals and evidence.
The existing policy engine decides ALLOW, REDACT, REVIEW or BLOCK. The
authorization service issues a request-bound, short-lived opaque token only
for ALLOW or a safely sanitized REDACT. The trusted tool adapter verifies and
consumes that token before invoking a side-effecting callback.

The new security property is: a guarded callback in the trusted adapter
cannot execute until the gateway has consumed one unexpired authorization for
the same request ID, agent/session, tool, operation, target, permissions, and
canonical payload. A second consumption fails. Gateway failure leaves the
callback uncalled.

The agent must **not** possess another path to the destination. Restrict its
network route, service credentials, filesystem permissions and direct tool
bindings separately. The adapter is application-level enforcement; it does
not transparently intercept arbitrary network packets or defend a process
that can bypass it. Tool rules and agent identities are trusted configuration,
not agent-controlled metadata. The current protocol does not authenticate the
human user or provide a human approval service. REVIEW holds the action.

On loopback, the prototype may run without a shared key. For network use,
configure a shared key and an HTTPS-terminating reverse proxy or trusted
private transport. The built-in Python HTTP server does not implement TLS.
Opaque tokens are random and stored only as hashes in one process; this is a
prototype, not an independently audited security appliance. Restart revokes
all outstanding tokens. Multiple gateway workers would need a shared atomic
store, which this milestone does not provide.

Single-use authorization is not end-to-end exactly-once delivery. If the
protected API executes and its response is lost, a **new** request ID could
repeat that side effect unless the protected API has its own idempotency key.

Model misclassifications remain possible. The protocol enforces the existing
policy decision; it does not prove that the policy decision is correct. No
physical UNO Q execution has been performed. Network isolation was tested
only in the local Docker harness.
