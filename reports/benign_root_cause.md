# Why E6 flags benign text (H1–H5)

Frozen E6 checkpoint, frozen DEV threshold 0.34, no training. Data: DEV splits of the new benign
sources (gold language tags) and the V5 DEV benign rows; deepset benign rows for analysis only
(they were already scored once as OOD; nothing here selects a model, threshold or training
setting on deepset). MASSIVE, the new OOD benign test, was not touched. Numbers:
`reports/benign_root_cause.json`.

| Hypothesis | Evidence | Verdict |
|---|---|---|
| H1 language | New DEV benign FPR: en 0.379 (n=1,694), de 0.798 (n=124), tr 0.978 (n=602). deepset benign by stopword language guess: de 0.886 (n=132), en 0.525 (n=179). The guess is 94% / 85% / 58% accurate for en / de / tr on gold-tagged rows. | **Supported** |
| H2 short input | English new DEV benign: ≤32 tokens 0.507 (n=1,157), 33–64 0.120, 65–94 0.055, 95–256 0.062. deepset English: ≤32 0.641, 33–64 0.034. | **Supported** |
| H3 security-vocabulary shortcut | Benign rows with "ignore", "system", "instruction", "prompt", "jailbreak", "security": FPR 0.03–0.17, lower than rows with none of the keywords (0.437). Deleting the keyword barely moves scores (e.g. ignore 0.135→0.128). | **Not supported** |
| H4 source/domain shift | English, ≤94 tokens: Dolly 0.469, deepset 0.525, Aya 0.385, OASST 0.326, but JailbreakLLMs regular prompts 0.123 and security docs 0.000. | **Supported** (source form, not topic) |
| H5 label semantics | deepset benign: 279/396 end with "?", 3 start with an imperative, 0 contain attack cues. No sign that deepset "benign" includes attack-like text. | **No evidence** |

Interpretation (not tested further here): in V5 v1 TRAIN, attacks are mostly short English game
inputs (Tensor Trust median 39 tokens) and benign text is mostly long forum prompts and
documentation. The model appears to use "short, non-English, not forum-like" as an attack cue.
Adding short, multilingual, human-written benign requests to TRAIN is the direct test of that.
