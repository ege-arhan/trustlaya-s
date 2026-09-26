# Authorization protocol v1

`POST /v1/authorize` receives a JSON object with exactly these fields:

```json
{
  "protocol_version": "1",
  "request_id": "unique-request-id",
  "timestamp": "2026-09-24T20:00:00+00:00",
  "agent_id": "demo-agent",
  "session_id": "demo-session",
  "tool": {"name": "external_api", "operation": "send", "target": "demo://sink"},
  "payload_sha256": "64-lowercase-hex-characters",
  "permissions": {"shell": false, "filesystem": false, "network": true,
                  "database": false, "email": false, "external_api": true,
                  "credential_access": false},
  "request": {"text": "Merhaba, toplantı yarın.", "arguments": {}}
}
```

`payload_sha256` is SHA-256 of UTF-8 JSON for the **entire** `request` object,
serialized with sorted keys, `,`/`:` separators and `ensure_ascii=False`.
IDs use ASCII letters, digits, `_` or `-` and have at most 64 characters.
Timestamps require a timezone and may differ from gateway time by at most 30
seconds. A configured tool rule must match name, operation, exact target,
agent ID and all seven permission flags. No client-supplied `human_approval`
flag is accepted. A duplicate request ID is denied.

The response contains `protocol_version`, `request_id`, `decision`,
`policy_version`, `reason_codes`, numeric `risk`, evidence **types and offsets**,
requested `tool`, `agent_id`, `session_id`, and, only when issued,
`decision_id`, `authorization_token`, `expires_at` and `authorized_request`.
The token is 32 random bytes encoded as URL-safe text; it is stored by SHA-256
digest on the gateway. It is an opaque capability, not a signed JWT. TTL
defaults to 15 seconds and is limited to 1–60 seconds. The server enforces
expiry using a monotonic clock.

For ALLOW, `authorized_request` is exactly the submitted request. For REDACT,
the server replaces concrete PII spans in `request.text`, leaves no arguments,
recomputes the payload hash, and re-runs the existing model and policy. A token
is issued only if that sanitized payload is ALLOW. A REDACT with no safe token
remains held. REVIEW and BLOCK never receive a token.

`POST /v1/consume` receives `authorization_token`, `decision_id` and the
**entire** `authorized_request`. It returns `{ "protocol_version": "1",
"valid": true/false, "request_id": "...", "decision_id": "...", "reason": "..." }`.
The gateway atomically checks token existence, expiry, consumed state,
decision ID and a digest of the complete authorized request, then marks the
token consumed. The guarded tool calls its side-effecting function only after
a matching `valid: true` response. A dropped consume response means no tool
execution, even if the gateway has already consumed the token.

The token prevents replay of the **same authorization**. It does not stop a
target from acting twice on two separate authorizations, so side-effecting
tools carry an idempotency key:

- `request.arguments.operation_id` (same ID rules as `request_id`) is bound by
  `payload_sha256` like every other field. It is not sent to the model and
  does not block REDACT; the sanitized request keeps it.
- The target stores `(agent_id, operation_id)` atomically. The same payload
  returns the earlier result (`output.replayed: true`); a different payload is
  rejected (`operation_id_conflict`). Retries must reuse the `operation_id`
  with a new `request_id`.

`POST /v1/tool` responses add `execution_status`:

| value | meaning |
|---|---|
| `executed` | the target confirmed the operation |
| `not_executed` | the request never reached a component that could execute it, or the target refused it |
| `unknown` | the request reached the adapter or target but no reply arrived; it may have executed. Retry with the same `operation_id`. |

`executed` stays a boolean and is `true` only for confirmed execution.

## Reading coverage

V2 reads `[CLS]` + the first 94 content tokens + `[SEP]` of the normalized
text (`head_94_v1`). `/analyze`, `/v1/authorize` and `/v1/tool` report
`coverage` (`total_tokens`, `read_tokens`, `truncated`, `read_spans`,
`unread_spans`, offsets in the normalized text) and `versions` (model file
SHA-256 prefix, policy hash, reading strategy). Regex evidence still scans the
full text. If the model did not read every token, an ALLOW or REDACT becomes
**REVIEW** with `incomplete_analysis_coverage` and no token is issued; a
missing coverage report counts as incomplete. BLOCK is never relaxed. The
2,000-character request limit is unchanged.

The current shared key protects the prototype HTTP endpoints from unrelated
clients when configured. It is not an agent identity proof. Do not expose
plain HTTP or the key/token to an untrusted network. Production deployment
needs authenticated TLS, hard process/network isolation and an audited shared
state design if multiple workers are required.
