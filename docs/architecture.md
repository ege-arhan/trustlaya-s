# Architecture

```text
User / AI Agent
  | text + optional permission metadata
  v
TrustLaya-S (42.1M parameter pretrained Turkish BERT)
  | shared mean-pooled representation
  +-- detection heads: PII, secret, prompt injection, dangerous instruction
  +-- risk heads: privacy, security, ethics, oversight, data governance
  +-- severity head + action head
  v
Temperature scaling + uncertainty estimate
  +-- regex and context-assisted evidence spans
  v
Deterministic policy engine (configs/policy.yaml)
  v
ALLOW / REDACT / REVIEW / BLOCK
  v
Future: Arduino UNO Q integration through an application gateway
```

All scores come from the model except `pii` and `secret`, which receive a conservative floor of 0.95 when deterministic evidence is detected. `model_action` is advisory. `action` is computed by policy. Agent permissions are evaluated by policy; they are not encoded into the student input in this MVP. Future UNO Q work needs a deployment gateway and device-specific profiling; no physical board was tested.
