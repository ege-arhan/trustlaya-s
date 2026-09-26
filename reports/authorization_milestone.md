# Fail-closed authorization milestone — 2026-09-24

The v2 model and existing policy engine were reused without retraining. A
versioned `/v1/authorize` request is validated against a trusted tool registry.
The gateway issues a 15-second opaque authorization only for ALLOW or a REDACT
whose sanitized payload passes reanalysis. `/v1/consume` atomically checks the
complete request binding and single-use state. The trusted tool adapter does
not call its fake side-effecting tool until consumption succeeds.

The seven-case real-ONNX demo executed successfully: benign ALLOW forwarded;
phone REDACT held because the sanitized payload did not pass reanalysis;
API-key BLOCK held; privileged prompt injection REVIEW held; first token use
forwarded; replay and changed payload were denied; gateway shutdown denied.
The separate deterministic protocol test proves a safely sanitized REDACT can
execute only the replacement text. No external tool was contacted.

MacBook local benchmark (`benchmarks/gateway.json`): 50 measured batch-one
requests plus one excluded warm-up, v2 FP32 ONNX CPU, fake no-op tool. Gateway
total p50/p95/p99 **5.568/5.939/6.051 ms**; authorization client
**5.219/5.521/5.656 ms**; model inference **4.507/4.794/4.926 ms**; policy
**0.003/0.004/0.004 ms**; consume **0.335/0.416/0.473 ms**. Zero timeouts
or denials occurred in the benign benchmark. These are not UNO Q measurements.

The full repository test suite passed **60/60** tests, including 34 new
authorization cases (counting parameterized cases). The existing model and
policy behavior remained unchanged; only timing fields were added to model
inference output.

The standalone CLI server was also started with the demo tool registry and a
local shared key. A guarded client allowed a benign fake send and held an
API-key send as BLOCK. This exercised the environment-based server startup
separately from the in-process demo.

Limitations: direct agent access to the protected target must be independently
disabled; the Python HTTP server has no TLS; the shared key is prototype-grade;
the token store is single-process and in-memory; human review approval is not
implemented. No physical UNO Q was tested. Model false positives/negatives and
calibration limitations are unchanged.
