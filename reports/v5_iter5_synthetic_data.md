# V5 iteration 5: synthetic data (TRAIN AUGMENTATION ONLY)

**None of these rows is gold, human-labeled, real-world or a test.** They are used only in TRAIN.
DEV, TEST1, TEST2 (deepset, Gandalf), MASSIVE test and MASSIVE dev contain no synthetic row (enforced
by assertions in the training and evaluation scripts). Raw synthetic text stays in git-ignored
`data/v5_iter5/`; per-row provenance (synthetic flag, method, original source and id, source/target
language, model and revision, timestamp, source checksum, question form, text hashes) is in
[`data/v5_iter5_manifest.json`](../data/v5_iter5_manifest.json).

| Synthetic source | Label | Rows | Method | Origin |
|---|---|---|---|---|
| tr_curated | ATTACK | 150 | author-curated synthetic Turkish (card) | 3nesdeniz/turkish-conversation-prompt-injection @29d7593, CC BY 4.0 |
| tr_curated | BENIGN | 600 | same | same (daily, technical and boundary rows) |
| mt_de_attack | ATTACK | 884 | OPUS-MT en→de (Helsinki-NLP/opus-mt-en-de @6183067, CC BY 4.0) | English TRAIN attacks (Tensor Trust, JailbreakLLMs, HackAPrompt) ≤ 96 words |
| mt_tr_attack | ATTACK | 884 | OPUS-MT en→tr (Helsinki-NLP/opus-mt-tc-big-en-tr @e539fc1, CC BY 4.0) | same English originals |
| mt_de_benign | BENIGN | 1100 | OPUS-MT en→de | Dolly TRAIN short questions (≤ 32 tokens) |
| mt_tr_benign | BENIGN | 998 | OPUS-MT en→tr | same |

Planned 1,100 / 1,000 attack translations; only 884 English TRAIN attacks met the sampling rule
(≤ 96 words, equal per source), so both languages use the same 884 originals.

**Synthetic attack share of attack TRAIN** (real attacks: 13018):
S1 Turkish 1034 (7.4%), S2 German 884 (6.4%), S3/S4 both 1918 (12.8%).
All below the 15% limit. Sampling mass for synthetic rows equals their count share within the
label, so source balancing does not inflate them. The 25% variant was not run.

**Leakage:** every synthetic row was clustered (MinHash, char 5-grams, Jaccard > 0.70) with all
TEST/OOD rows; 2 rows were dropped
(mt_tr_benign|massive: 1, mt_tr_benign|massive_dev: 1). English originals
come only from TRAIN clusters, so no translated attack is a translation of a TEST row. No real
Turkish/German public attack test exists to check against (iteration 4).

**Known artifact (found by the audit before any TEST result):** translated attacks are rarely
questions (≈5%), while the translated benign controls are mostly questions (≈90%), because the
benign controls were sampled as short questions by plan. This can teach "question = benign",
which works against deepset's question-shaped attacks. S4 and S5 carry this risk most. See
[translation audit](v5_iter5_translation_audit.md).
