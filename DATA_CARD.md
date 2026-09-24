# TrustLaya-S Advanced data card

## Purpose and composition

The v1 training corpus is [TrustLaya-S-synthetic](https://huggingface.co/datasets/ege-arhan/TrustLaya-S-synthetic), 10,000 controlled synthetic rows. Local train/validation/test sizes are 6,972/1,053/1,975. The full [audit](reports/dataset_audit.md) found only 2,682 unique texts and 143 normalized text prototypes. Train has Turkish and English; validation is English-labeled only; test is mixed-labeled only. This split is unsuitable for language-specific generalization claims.

The v2 PII candidate updates **only the PII risk-head row**. It uses 3,653 scenario-grouped training rows from the MIT-licensed, synthetic/curated [Turkish Privacy Filter Dataset](https://huggingface.co/datasets/yusuf-said/turkish-privacy-filter-dataset), plus 1,000 synthetic v1 training anchors. Development has 478 rows from `json_log` and `chat_transcript`; the held-out scenario set has 979 rows from `ocr_scan`, `server_log`, and `call_center_log`. Scenario names and row counts are in `reports/advanced_pii_experiment.json`. The source annotations are not independent human adjudication. Supported positive categories are account number, person, phone, email, and address; dates and URLs are excluded from this task mapping.

The evaluation scripts pin upstream revisions and record source hashes in their JSON reports. The training script writes new candidates under `models/candidates/` by default and refuses to overwrite released checkpoints.

## Diagnostic and test sources

| Source | License | Use | Redistribution |
|---|---|---|---|
| [deepset prompt injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | v1 external injection diagnostic; earlier rejected fine-tune experiment | No source rows in this repo |
| [BTX24 Turkish Privacy PII NER](https://huggingface.co/datasets/BTX24/turkish-privacy-pii-ner) | CC BY 4.0 | Unused-for-training 2,000-row stratified v2 PII diagnostic, 1,000 positive and 1,000 task-specific negative | Aggregate statistics only |
| [AgentInjectionBench](https://huggingface.co/datasets/ppradyoth/AgentInjectionBench) | Apache 2.0 | Separate 182-case agentic prompt-injection diagnostic with 142 attacks and 40 matched benign controls; measures text detection on tool results, not agent execution | Aggregate statistics only |
| [BPI-Guard Dataset](https://huggingface.co/datasets/MelikeErdogan/bpi-guard-dataset) | MIT | Experimental injection-head candidates: 17,037 train, 3,020 validation, 2,961 test after 12 direct/original-text overlap exclusions; labels combine prompt injection, jailbreak and other adversarial content | No source rows redistributed |
| [zachz Prompt Injection Benchmark](https://huggingface.co/datasets/zachz/prompt-injection-benchmark) | MIT | 299-row cross-source check of the first BPI-only candidate after four exact-training-overlap exclusions; subsequently inspected, so not fresh for later candidates | Aggregate statistics only |
| [PolyGuardBench](https://huggingface.co/datasets/fevziegeyurtsevenler/PolyGuardBench) | CC BY 4.0 | One-shot cross-source check of frozen injection candidates: 190 prompt-injection attacks versus 360 benign over-refusal examples; report these axes separately | Aggregate statistics only; attribute Fevzi Ege Yurtsevenler |
| [Prowl secrets corpus](https://huggingface.co/datasets/Podric/prowl-secrets-corpus) | CC BY-NC 4.0; derived portions retain upstream terms | v1 diagnostic only; **not used for training v2** | No source rows or spans |
| [Rogue Security hard benign cases](https://huggingface.co/datasets/rogue-security/real-world-benign-use-cases) | License marked `other` | Diagnostic only | No source rows |

`TrustLaya-Adversarial` contains 24 researcher-authored clean prompts and 216 controlled transformations, half injection, half benign. `TrustLaya-TR-Benchmark` contains only 33 researcher-authored smoke cases (11 per language). Both are in `data/benchmarks/`; neither is human verified or a production-representative sample. Their scores must be presented with sample size and provenance.

## Leakage and labeling cautions

V1 template families do not cross original splits, and no exact or normalized text crosses them under the audit normalizer. This does **not** rule out semantic paraphrase leakage. The v2 privacy scenarios are disjoint across train/development/test, and their normalized texts had zero overlap under the same normalizer. Source-specific formulas, vocabulary, or generation style may still overlap. The v2 scenario test was inspected for promotion; future tuning against it invalidates it as a fresh holdout. The BTX24 test was inspected after model training and has the same limitation for future revisions.

The BPI source labels include jailbreak and social engineering as well as prompt injection, so its F1 is not the same task as a strict tool-output injection detector. Its test includes generated variants and may retain family-level leakage beyond the 12 direct/original-text overlaps removed here. The original synthetic validation is English-labeled only and proved too easy: the released v2 injection head scores F1 1.0 there but 0.555 on the mixed-only synthetic test. The new joint candidate used only training and validation splits for fitting and choice; its PolyGuardBench check was one-shot, while BPI, original synthetic and AgentInjectionBench tests were already inspected and serve as regressions.

No actual credentials or real personal data are intentionally published in our generated benchmark files. Public source datasets should be fetched from their own repositories with their stated licenses. Do not treat teacher probabilities or generated labels as ground truth.
