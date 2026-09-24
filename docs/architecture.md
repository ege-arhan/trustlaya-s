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
  +-- permission and session-chain analysis
  +-- transparent rule-constrained risk fusion
  v
Deterministic policy engine (configs/policy.yaml)
  v
ALLOW / REDACT / REVIEW / BLOCK
  v
Future: Arduino UNO Q integration through an application gateway
```

The v2 candidate changes only the PII head: a logistic regression head is trained on frozen v1 encoder features from Turkish privacy scenarios. Other heads remain v1. All scores come from the model except `pii` and `secret`, which receive a conservative floor of 0.95 when deterministic evidence is detected. `model_action` is advisory. `action` is computed by policy. `raw_scores` and `calibrated_scores` distinguish the signal stages; fused scores are risk indices, not probabilities. Agent permissions and bounded session event chains are evaluated by deterministic policy; they are not encoded into the student input. Future UNO Q work needs device profiling; no physical board was tested. See [edge deployment](edge_deployment.md).
