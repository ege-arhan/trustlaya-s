# V5 Jev-only SILVER annotation run

**Date:** 2026-09-25. **Status:** complete automatic annotation experiment, **not GOLD**. No human labels were created or used to adjudicate this run. No V5 model was trained, published, or connected to the gateway. V2 remains the default; V3 and V4 remain NO-GO.

## Corpus and rights

This is a **new 600-row corpus**, not a relabeling of all rows in the frozen 600-row human-review packet. The original packet and its checksums remain unchanged.

| Source | Rows | Selection | License information |
|---|---:|---|---|
| [OWASP Cheat Sheet Series](https://owasp.org/projects/cheat-sheet-series) | 79 | All permitted rows in frozen packet | CC BY-SA 4.0 |
| [Microsoft MCP security material](https://github.com/microsoft/mcp-for-beginners) | 92 | All permitted rows in frozen packet | MIT repository |
| [Stack Exchange](https://stackoverflow.com/help/licensing) | 89 | All permitted rows in frozen packet | CC BY-SA 4.0, author/post URLs retained |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections), revision `4f61ecb038e9c3fb77e21034b22511b523772cdd` | 340 | Deterministic seed 5025; 170 source-label 0 and 170 source-label 1 from **train** only | Card says Apache-2.0; nested metadata says CC BY 4.0. Both declarations are recorded; conflict remains unresolved. |

The original packet's 340 [Tensor Trust](https://github.com/HumanCompatibleAI/tensor-trust-data/issues/3) candidates were **not sent to Vercel** because the data repository lacks an explicit license. The replacement deepset rows are separate records with separate IDs and checksums; the official deepset *test* partition was untouched. Normalized exact text duplicates among selected rows were removed before selection. The deepset train partition has appeared in earlier TrustLaya-S experimentation, so this SILVER corpus is **not an independent hidden test**.

## Annotation protocol

The same eight typed [Vercel AI Gateway evaluate](https://vercel.com/docs/ai-gateway/modalities/evaluation) questions went to `typesafe-ai/jev` for each row: intent (`ATTACK`, `BENIGN_DUAL_USE`, `NORMAL`, `UNRESOLVED`), primary attack vector, benign subtype, directed attack, quoted reference, actual PII, actual secret, and obfuscation. Responses and probabilities were stored exactly as returned. The projection is marked `JEV_SILVER`. `UNRESOLVED` remains a distinct label. No human GOLD, threshold fitting, calibration, or accuracy/F1 score was generated.

The model-facing state contained the source text, but not the source name or source proxy label. The question schema SHA-256 is in the [manifest](../benchmarks/v5/jev_silver_manifest.json). The model alias was returned; a stable underlying model build identifier was not provided in these responses. The request is vulnerable to adversarial wording in the evaluated state, and TypeSafe [documents Jev's primarily English support](https://docs.typesafe.ai/models); this run does not establish Turkish performance.

## Results: label distribution, not accuracy

| Source | N | ATTACK | BENIGN_DUAL_USE | NORMAL | UNRESOLVED |
|---|---:|---:|---:|---:|---:|
| OWASP | 79 | 7 | 41 | 31 | 0 |
| Microsoft | 92 | 0 | 45 | 47 | 0 |
| Stack Exchange | 89 | 0 | 9 | 80 | 0 |
| deepset train | 340 | 88 | 6 | 243 | 3 |
| **Total** | **600** | **95** | **101** | **401** | **3** |

deepset's **source proxy labels** and Jev's SILVER intent disagree materially. Among 170 source-label-1 rows, Jev chose `ATTACK` 88 times, `NORMAL` 75, `BENIGN_DUAL_USE` 6, and `UNRESOLVED` once. Among 170 source-label-0 rows, Jev chose `NORMAL` 168 times and `UNRESOLVED` twice. Exact non-ambiguous projection agreement is **256/340**. These source labels are not a separately reviewed Jev ground truth; this is a discordance diagnostic, **not Jev recall, false-positive rate, or accuracy**. The 82 source-label-1 rows not marked `ATTACK` are a serious reason to avoid training or releasing from these labels without further independent evidence.

Quality flags: 164 rows had maximum intent-choice probability below 0.80; 16 Jev `ATTACK` choices also had quoted-reference probability at least 0.5; 2 `ATTACK` choices had directed-attack probability below 0.5. These are internal consistency checks, not correctness proofs.

## Context and run measurements

The V2 tokenizer reports maximum sequence length 512, but the **default V2 `Analyzer` uses `max_length=96`**, leaving approximately **94 content tokens** after special tokens. Counting tokens after the actual V2 text normalization, this corpus has median length 61 tokens and p95 529. **205/600** inputs exceed the effective 94-token head window. **32/600** also exceed the tokenizer's 510-content-token architecture window; Jev labeled only 2 of those 32 `ATTACK`. The replacement therefore **does not resolve the long-context benchmark requirement**; the excluded Tensor Trust rows contained many long attacks. Length bucket counts and hashes are in the machine-readable manifest.

Observed end-to-end per-call client latency, including transient `503` retries: p50 **2,103 ms**, p95 **34,310 ms**, p99 **94,043 ms**. Vercel reported 562,666 input tokens, 143,854 output tokens and gateway cost `0` for this run; that reported cost does not predict future billing. The first concurrent pass saved 570/600 responses; 30 transient `503` failures completed in a single-worker retry. Final count is **600 unique IDs, 600 matching source-text hashes, 0 remaining failures**. Latency is neither Jev model-only nor Arduino UNO Q latency.

## Artifacts and reproduction

Tracked: [runner](../scripts/run_v5_jev_silver.py), [summarizer](../scripts/summarize_v5_jev_silver.py), [manifest](../benchmarks/v5/jev_silver_manifest.json), [tests](../tests/test_v5_jev_silver.py). Local gitignored: `benchmarks/v5/private/jev_silver_600.jsonl` (typed answers, hashes, usage), `jev_silver_600_with_text.jsonl` (source text joined to SILVER labels), and `jev_replacement_340.jsonl` (selected replacement packet). The private joined file is mode 0600. It includes source URLs and license fields; keep it local until the deepset license metadata conflict and redistribution terms are resolved. No API credential is in either output; the temporary local credential file was removed after completion.

```bash
export AI_GATEWAY_API_KEY='your-key'
.venv/bin/python scripts/run_v5_jev_silver.py --workers 1
.venv/bin/python scripts/summarize_v5_jev_silver.py
.venv/bin/python -m pytest -q tests/test_v5_jev_silver.py
```

The runner is resumable and checks selected IDs and text hashes. The summarizer refuses partial results and writes the public aggregate manifest plus a private joined SILVER file. The frozen original packet is not changed.

## Decision

**SILVER DATASET COMPLETE; MODEL PROMOTION NO-GO.** Jev performed all annotations without human labeling, as requested. Its disagreement with the existing deepset source labels and this corpus's weak long-context coverage mean it cannot provide a validated V5 benchmark or support a TrustLaya-S release. The next research step is a separately sourced, license-clear, long-form attack and benign corpus with an independent evaluation standard. If human annotation remains excluded, report only source-proxy and cross-model agreement diagnostics; do not call them GOLD or external accuracy.
