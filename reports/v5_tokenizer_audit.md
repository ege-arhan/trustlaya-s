# Tokenizer length audit

The current tokenizer belongs to the Turkish-first `ytu-ce-cosmos/turkish-medium-bert-uncased` backbone. We measured the same **1,168 unlabeled V5 candidate texts** with the frozen TrustLaya tokenizer and a pinned, Apache-2.0 [multilingual BERT tokenizer](https://huggingface.co/google-bert/bert-base-multilingual-cased) at revision `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`. Only tokenization was compared; no weights were changed. [Machine results](../benchmarks/v5/tokenizer_audit.json).

| Source candidate family | N | Current tokens/word | Multilingual reference tokens/word | Current >94 | Reference >94 |
|---|---:|---:|---:|---:|---:|
| Microsoft MCP security text | 92 | 2.924 | 2.163 | 58 | 42 |
| OWASP security text | 79 | 2.852 | 2.078 | 28 | 13 |
| Stack Exchange questions | 89 | 2.625 | 2.068 | 87 | 79 |
| Tensor Trust game attempts | 908 | 1.994 | 1.632 | 439 | 338 |

The documentation/forum sources are predominantly English; the game source's per-row language is **undetermined**. Word counts are Unicode word-boundary counts, so code and formatting can inflate the ratios. On these samples the Turkish tokenizer uses more subwords than the multilingual reference, which supports a **length-efficiency** concern. It does **not** prove a classification or calibration cause. A tokenizer cannot simply be swapped under frozen BERT weights: token IDs would refer to different embeddings. There is no independent, human-labeled Turkish intent benchmark in this audit.
