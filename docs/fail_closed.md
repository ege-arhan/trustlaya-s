# Fail-closed behavior

The trusted `GuardedTool` calls a protected sender only after a valid consume
response. Every error below produces `executed: false`:

| Condition | Result |
|---|---|
| Gateway absent, crashed, timed out or connection dropped | No sender call |
| Malformed, oversized or unknown gateway response | No sender call |
| BLOCK or REVIEW | No authorization; no sender call |
| REDACT without a safe, rechecked sanitized request | No sender call |
| Missing, empty, altered, copied, expired or replayed token | Consume invalid or local rejection |
| Different request ID, agent/session, tool, operation, target, permissions or payload | Local rejection or gateway digest mismatch |
| Expired timestamp or duplicate request ID | Authorization rejected |
| Consume response lost after server marks token spent | No sender call; retry sees replay denial |

The gateway stores hashed tokens and a bounded request-ID set in memory.
Restart revokes tokens. The authorization audit records identifiers, decisions,
numeric risk summary, issuance and consumption status. It omits raw request
text, arguments, token and evidence text. Target names in the trusted rule
configuration should be identifiers without secrets.

`REVIEW` is a hold; setting `human_approval` in the request cannot bypass it.
A future approval component needs its own authenticated flow and a fresh
authorization. The legacy `/analyze` API and `TrustGateway` helper remain for
backward compatibility but are **not** the new guarded-tool protocol. Use
`GuardedTool` for side-effecting operations.
