# Security boundary

```text
Agent (no direct target access)
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

Model misclassifications remain possible. The protocol enforces the existing
policy decision; it does not prove that the policy decision is correct. No
physical UNO Q execution or network isolation test has been performed.
