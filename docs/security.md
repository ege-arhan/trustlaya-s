# Security notes

This prototype makes a risk estimate, extracts evidence, and then applies deterministic policy. A model output cannot override the secret or PII transfer rule. Policy thresholds can be changed in `configs/policy.yaml`; a real deployment needs per-organization review, audit logging, access controls and feedback loops. Evidence text may itself be sensitive, so production logging should redact it. Agent metadata is untrusted input and should be supplied from trusted capability configuration rather than a user's free text.

Regex detectors validate Turkish national ID checksums, credit-card Luhn checksums and IP syntax. Their scope is limited. Name/address detection relies on field labels. Prompt injection protection is incomplete; zero-day or obfuscated attacks can evade the model. Do not use this prototype as the sole approval mechanism for high-impact agent actions.
