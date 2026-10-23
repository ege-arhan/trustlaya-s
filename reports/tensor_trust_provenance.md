# Tensor Trust provenance and label semantics

## Source

| Item | Value |
|---|---|
| Data repository | https://github.com/HumanCompatibleAI/tensor-trust-data |
| Pinned revision | `747a75e096761ebc01bd3970158827326b4add23` (last push 2024-03-17) |
| Pipeline code (semantics reference) | https://github.com/HumanCompatibleAI/tensor-trust @ `f0b055451415e23d444703fbb8a47daf1d4b2c3c`, `data-pipeline/src/ttdata/` |
| Paper | Toyer et al., *Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game*, arXiv:2311.01011 |
| Files used | v1 hijacking and extraction robustness benchmarks, v1 extraction-detection set, raw dumps v1 (2023-09-10, 126,808 attacks) and v2 (2024-02-16, 563,349 attacks) |
| Checksums | `data/v5_source_pins.json` (SHA-256 per file); mismatch fails `scripts/fetch_tensor_trust.py` |
| License | **The data repository has no LICENSE file.** The game/pipeline code repository is BSD-2-Clause. No data license is assumed from the code license. |
| Storage | Raw files only in git-ignored `data/external/tensor_trust/<revision>/`. No raw attack text is committed or copied into reports; tracked files hold hashes, labels, splits and counts. |

## How labels are derived

The raw dump records game events, not "attack / benign" labels. An attack's
success is judged by the defender's LLM output, so the same text can succeed
against one defense and fail against another. `src/trustlaya/tensor_trust.py`
reproduces the upstream candidate heuristics (`derived_features.py`):

- **Hijacking candidate:** the input was answered with "access granted"
  (medium regex) against some account where it was not the access code, was
  used against at least two accounts, and has more than one word or more than
  30 characters.
- **Extraction candidate:** the LLM output contains the access code, or the
  input is the attacker's last unsuccessful attack before they entered the
  correct access code on another player's account (trajectory back-chaining).

Upstream then deduplicated and adversarially filtered candidates against
several LLMs to build the v1 benchmarks. That LLM filtering is not rerun here;
rows that appear in the published v1 benchmarks carry the stronger
`*_benchmark_v1` category.

Parser check on the v1 dump: the reproduced heuristics recover **490/492**
unique hijacking-benchmark attacks and **438/438** extraction-benchmark
attacks (v2 dump: 491/492, 438/438). Unexpected keys or types raise
`SchemaError` and stop the build.

| Native category | Training label | Unique texts (v2 dump) |
|---|---|---|
| `hijacking_benchmark_v1` | ATTACK | 492 |
| `extraction_benchmark_v1` | ATTACK | 438 |
| `hijacking_candidate` | ATTACK | 7,141 |
| `extraction_candidate` | ATTACK | 5,771 |
| `unverified_attempt` (no success evidence) | EXCLUDED, not benign | 228,847 |
| `self_or_sandbox_attack` | EXCLUDED | 50,724 |
| `access_code_entry` (a password) | EXCLUDED | 1,220 |
| `tos_removed_text` | EXCLUDED | 43 |
| defenses (pre/post prompts) | EXCLUDED, not a normal-user distribution | — |

After within-source normalized-exact deduplication, 11,994 Tensor Trust attack
texts remain. Near-duplicate clusters (MinHash, Jaccard > 0.70, across all
sources) are split as units: TRAIN 7,052 / DEV 1,346 / TEST 3,596. Every
cluster containing a v1 benchmark attack is forced into TEST, so the upstream
benchmark stays out of training.

Tensor Trust contributes **no benign rows**. Benign text comes from other
real sources (JailbreakLLMs regular prompts, security documentation and
forum prose, arXiv abstracts).
