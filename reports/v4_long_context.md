# V4 long-context diagnosis and read strategies

The measured [token-length diagnostic](long_context_diagnostic.md) used the frozen JailbreakLLMs corpus before V4 fitting. Encoder limit: 512 total / 510 content tokens. Existing v2/v3 application limit: 96 total / 94 content tokens. Clean attacks: 635 rows, median 540 content tokens, p95 1628; 600 (94.5%) exceeded the actual application read. A lexical cue's first match occurred after token 94 in 52 attack rows; this is a heuristic, not a gold attack span.

After disabling **both** the tokenizer backend's implicit truncation and padding, six strategies were compared on a source-separated proxy DEV of 340 long Bordair live-game attack attempts and 105 OWASP explanatory paragraphs. The near-duplicate `bordair:13198` was excluded before final selection. All model weights were fixed for this comparison; v3 uses its prior calibration/0.30 threshold and v4 below is **raw at 0.50**. No final JLL scores selected the strategy.

| Read strategy | v3 F1 | v3 recall | v4 raw F1 | v4 raw recall | v4 raw FPR | Mean windows/doc |
|---|---:|---:|---:|---:|---:|---:|
| HEAD | 0.012 | 0.006 | 0.204 | 0.115 | 0.029 | 1.0 |
| TAIL | 0.012 | 0.006 | 0.095 | 0.050 | 0.010 | 1.0 |
| HEAD_TAIL | 0.012 | 0.006 | 0.157 | 0.085 | 0.000 | 1.0 |
| SLIDING | 0.006 | 0.003 | 0.046 | 0.024 | 0.000 | 5.2 |
| WINDOW_MAX | 0.063 | 0.032 | 0.422 | 0.271 | 0.038 | 5.2 |
| WINDOW_LOGIT_POOL | 0.006 | 0.003 | 0.046 | 0.024 | 0.000 | 5.2 |

WINDOW_MAX led this proxy DEV on F1. Its benefit carries a multiple-comparisons cost: on frozen JLL test, v4 FPR was 0.860 for >510-token prompts versus 0.294 for <=94-token prompts. A longer document supplies more opportunities for one high false alarm. The frozen v3 head remained low-recall across all six strategies; truncation was not its sole failure.

The strategy experiment's batched encoder work was 3211 ms for 2738 unique windows; this is an aggregate DEV computation, not batch=1 latency. [Local warm latency](v4_latency.json): 30 documents, 10 per length bin, includes tokenizer-adjacent Python/encoder work, excludes network and policy. MPS p50: 4.4, 13.1, 22.3 ms by increasing length. First cold inference 88.5 ms, model load 643.9 ms. These are Mac measurements, not UNO Q.
