# V3 external-data protocol (fixed before v3 training)

## Task definitions and sources

| Role | Source / license / pinned version | Use | Caveat |
|---|---|---|---|
| PII train/dev/test | [TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark), MIT, `558e09e26d6b36f5f78440074e6a233946d98bd9` | Official ECHR train/dev/test case partitions | Only 328/1,014 train cases have `quality_checked` gold. Dev: 127/127; test: 127/127. |
| Attack train/dev/diagnostic test | [Lakera Gandalf ignore instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions), MIT, `04737b65e90a6794ec227012e4a255a7def6344b` | Official 777/111/112 human-submitted game-prompt partitions | Embedding-similarity filtered, not individually verified. Weak labels; prompt injection, not general jailbreak. |
| Benign train/dev | [prompts.chat](https://github.com/f/prompts.chat), prompt data CC0, snapshot `f78a1c5136fa080155d928e0d7e2b4a41ddef03e`, CSV SHA-256 `c506bbf29106058a021e5cf85271bb97c9856c2b7fcc9f337421cdc8b00964c6` | Human/community-submitted ordinary prompts; contributor-disjoint split | No native benign/attack labels; conservative exclusion of obvious attack instructions and manual audit required. Some roleplay or security prompts remain ambiguous. |
| Final external direct-jailbreak test | [JailbreakLLMs](https://github.com/TrustAIRLab/JailbreakLLMs), MIT, `2dbd7bbc25f1b156552678f451bddbc787cd679f` | Frozen 5,888-row community subset, 636 attack and 5,252 regular | Distinct source; direct jailbreak is not equivalent to Gandalf prompt injection. No v3 tuning on it. |
| Final external PII test | TAB official test | Frozen 2,079 uncontaminated windows | No v3 training or tuning. |

TAB `entity_mentions` are **character spans** with `entity_type` and `identifier_type`, not binary document labels. The frozen v2 comparison projected only fully contained `DIRECT PERSON/CODE` spans to binary 94-token-window presence. V2's 0.107 F1 is **window presence F1**, not entity F1. V3 retains that projection for paired comparison and separately reports token, exact span and document metrics. Its BIO labels are `O`, `B/I-PERSON`, `B/I-CODE`; other TAB categories are deliberately outside this narrow task and must not be silently treated as evaluated PII categories.

The current encoder is frozen for the first controlled experiment. A lightweight token head and a distinct attack head are trained on external data, preserving every v2 head. Development partitions select thresholds and calibration; final tests are read only for the last evaluation. No synthetic TrustLaya-S examples are used for v3 parameter fitting. Existing synthetic checks remain internal regression tests.

No Turkish human-annotated external test has yet been identified; Turkish performance cannot be inferred from the English TAB or Gandalf/JailbreakLLMs corpora. Tensor Trust has uncertain dataset reuse rights and is excluded. WildJailbreak is generated data and is excluded from primary training. The original `jackhhao/jailbreak-classification` source overlaps the frozen JailbreakLLMs test and is excluded.

The official HackAPrompt parquet is MIT but gated and could not be accessed in this run (HTTP 401). The much larger Pliny copy was not acquired; it is not interchangeable with the official competition release. ScaleAI MHJ is gated and CC-BY-NC; it was not used. No inaccessible source is counted as v3 training data.
