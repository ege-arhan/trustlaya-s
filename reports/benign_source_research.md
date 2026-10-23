# Public benign sources for V5 iteration 2

All sources are fetched at pinned revisions with SHA-256 checks (`data/v5_source_pins.json`)
into git-ignored `data/external/`. Raw text is not committed. Labels are source-native: every row
below is a user-written request or question in a dataset built for ordinary assistant use. No row
was annotated by us or by an LLM. "Benign" here means *not a prompt-injection/jailbreak attempt by
source construction*; it is not a claim that every row is harmless in every sense.

| Source | Official source / revision | License (evidence) | Languages used | Rows used | Provenance | Role |
|---|---|---|---|---|---|---|
| Dolly 15k | `databricks/databricks-dolly-15k` @ `bdd27f4` | CC BY-SA 3.0 (dataset card `license: cc-by-sa-3.0`) | en | 4,200 (cap) | Databricks employees wrote instructions; we use only the `instruction` field | TRAIN/DEV/TEST |
| OpenAssistant oasst2 | `OpenAssistant/oasst2` @ `179dd21`, file `2023-11-05_oasst2_ready.messages.jsonl.gz` | Apache-2.0 (card `license: apache-2.0`) | en, de, tr | 4,786 | Volunteer-written first user turn; filtered to review_result=true, not deleted, not synthetic, spam label < 0.5 | TRAIN/DEV/TEST |
| Aya dataset | `CohereLabs/aya_dataset` @ `f9ea045` | Apache-2.0 (card `license: apache-2.0`) | en, de, tr | 7,638 | Aya annotators wrote or re-annotated prompts (`inputs`) | TRAIN/DEV/TEST |
| MASSIVE 1.1 | `alexa/massive`, S3 release tarball (ETag `51e0da2a…-3`), test partition | CC BY 4.0 (`1.1/LICENSE` in the tarball; HF card `cc-by-4.0`) | en-US, de-DE, tr-TR | 2,978 (1,000 sampled per locale, 22 removed as pool overlaps) | English crowd-written voice-assistant requests; German and Turkish are professional localizations of them | **TEST2_OOD_BENIGN only** |

Why these:
- Short natural requests and questions are exactly the distribution where E6 fails (see
  [benign_root_cause.md](benign_root_cause.md)); Dolly, OASST and Aya are human-written and
  instruction-shaped, not forum jailbreak discussions.
- OASST and Aya are the only permissively licensed human sources found with German and Turkish
  user prompts. German stays small: 544 TRAIN rows (OASST 401, Aya 143). Turkish comes almost only
  from Aya (2,865 TRAIN rows; OASST has 9 Turkish first prompts).
- MASSIVE is a different source and genre (voice-assistant commands) in all three languages, so it
  tests whether benign improvements transfer beyond the training sources and beyond deepset. Its
  German/Turkish rows are translations; results on them are reported as such.

Rejected or deferred:
- **deepset/prompt-injections**: stays OOD test only (rule of this iteration).
- **GermanQuAD / GermanDPR** (German questions): published by deepset, the same organization as
  the OOD test; excluded to keep the OOD test independent.
- **Alpaca / self-instruct style sets**: model-generated text, not human-written.
- **HackAPrompt**: gated; terms not accepted by the account owner.
- **Security Stack Exchange questions**: good security-vocabulary hard negatives, not fetched in this
  iteration. The pool therefore has only 246 new TRAIN rows with security vocabulary, plus the
  existing security documentation (503) and arXiv abstracts (392).

Checks performed:
- Exact normalized duplicates dropped within each source.
- MinHash near-duplicate clusters (char 5-grams, Jaccard > 0.70) computed jointly with every V5 row,
  deepset, Gandalf and MASSIVE. New rows that cluster with an attack (33), with an OOD row (10) or
  that would bridge two V5 splits are excluded; MASSIVE rows that cluster with any pool row (8) are
  dropped from the OOD test.
- Caps fixed before training: 3,000 TRAIN / 600 DEV / 600 TEST rows per (source, language).
- Counts, length buckets and security-vocabulary counts: `reports/benign_repair_dataset.json`.
