# V4 data provenance and leakage screen

| Role | Source / pinned revision | N used | Label origin / caveat | License |
|---|---|---:|---|---|
| TRAIN positive | [JailbreaksOverTime](https://github.com/wagner-group/JailbreaksOverTime) `94a2e998282301d545f92177e3fff8aab11fb0dd` / JailbreakChat compositions | 95 | Human-authored jailbreak templates combined with sampled harmful payloads; composed examples, **not wholly raw human prompts** | MIT repository; original source attribution retained locally |
| TRAIN negative | Same release / WildChat human user prompts | 2078 | Real user prompts but source filtering is not per-row human verification; WildChat original [ODC-BY](https://huggingface.co/datasets/allenai/WildChat-1M) attribution applies | ODC-BY underlying |
| DEV attack | [Bordair live game](https://huggingface.co/datasets/Bordair/bordair-multimodal) `398e3f875ffd32ec2e6817a95feebfcbae41643a` | 340 | Human game submissions >=500 characters, label by context/construction; goal is password/rule hijack, not general harmful jailbreak | MIT |
| DEV benign | [OWASP LLM Top 10 2026](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10) `9253e38ade58e959b531c0c5c9a4842272c9cd0e` | 105 | Human-written explanatory paragraphs, not ordinary user requests | CC BY-SA 4.0 |
| Final direct-jailbreak test | [JailbreakLLMs](https://github.com/TrustAIRLab/JailbreakLLMs) `2dbd7bbc25f1b156552678f451bddbc787cd679f` | 5,761 | Previously frozen v2/v3 clean paired cohort; unseen by V4 fitting and selection, **not a never-before-examined research source** | MIT |
| Additional untouched-source test | [deepset prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) official test, `4f61ecb038e9c3fb77e21034b22511b523772cdd` | 116 | Independent binary prompt-injection labels, sparse annotation provenance and a related but different task | Apache-2.0 |

Training excluded all 848 JOT records sourced from JailbreakLLMs. Its 154 template-prefix groups were capped at four examples each before screening. Exact matches removed 8; normalized char 4-5-gram cosine >=0.85 removed 488 additional TRAIN examples against the frozen JLL set. One near-matching Bordair DEV example was removed before strategy/threshold reselection. Final cross-role audit: TRAIN/DEV 0, DEV/TEST 0, TRAIN/TEST 0 near matches. Raw source text and row predictions remain gitignored; reports store counts, hashes and provenance.

No verified independent Turkish test was available in these acquisitions. Language-specific Turkish, English and mixed scores are **not estimated** from unannotated language fields.
