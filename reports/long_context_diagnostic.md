# Long-context diagnostic before v4 training

This diagnostic used the frozen JailbreakLLMs community rows (full N=5,888; clean paired N=5,761), the exact v2/v3 tokenizer, and the existing `normalize` function. No model was trained or threshold chosen. The [machine-readable metrics](v4_long_context_metrics.json) and gitignored per-row audit `benchmarks/external/predictions/v4_length_audit.json` contain hashes, lengths and cue positions, **not raw prompt text**.

## Actual limits

The BERT encoder has `max_position_embeddings=512`, and the tokenizer declares a **512 total-token** maximum. Two special tokens leave **510 content tokens** for a valid single pass. The current v2 `Analyzer.analyze()` and v3 head extraction both set `max_length=96`, so they expose only the **first 94 content tokens**. This 96-token application choice, not the architecture, is the current external-evaluation truncation boundary.

| Clean test class | N | Median chars | Median tokens | p90 tokens | p95 tokens | p99 tokens | >94 tokens/currently truncated | >510/model single-pass capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attack | 635 | 1565 | 540 | 1436 | 1628 | 2276 | 600 (94.5%) | 337 (53.1%) |
| Regular | 5,126 | 662 | 223 | 824 | 1199 | 2417 | 4,053 (79.1%) | 1,138 (22.2%) |

In the clean positive set, 381 prompts contain at least one regex cue for override, system-prompt disclosure, role change or policy bypass. In **52** of these, the *first* matched cue begins at or beyond token 94, so the current head-only evaluation cannot see that matched phrase. 182 have at least one matched cue beyond 94. At the physical 510-token limit, 14 have the first cue beyond a valid single pass. The regex is a lexical heuristic: it misses indirect/obfuscated attacks and can match quoted text. These counts are **not gold attack-span recall**.

**Answer:** the current classifier sees at most the first 94 content tokens per prompt. It truncates 94.5% of attacks and 79.1% of regular prompts. A 512-token single pass would still truncate 53.1% of attacks. The next experiment must compare head, tail and windowed inference on development data before changing weights.
