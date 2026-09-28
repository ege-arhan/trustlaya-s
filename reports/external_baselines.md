# Compatible external baselines

Same retained sample IDs, same gold, fixed 0.50 score threshold. A baseline score is a model or recognizer score, not a calibrated risk probability.

| Task | System | N | Precision | Recall | F1 | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| TAB narrow PII | TrustLaya-S v2 frozen | 2079 | 0.120 | 0.096 | 0.107 | 0.083 | 0.904 |
| Jailbreak proxy | TrustLaya-S v2 frozen | 5888 | 0.113 | 0.945 | 0.202 | 0.900 | 0.055 |
| TAB narrow PII | Presidio Analyzer 2.2.364 + spaCy en_core_web_lg 3.8.0 | 2079 | 0.336 | 0.954 | 0.497 | 0.222 | 0.046 |
| Jailbreak classification | jackhhao/jailbreak-classifier | 5888 | 0.121 | 0.970 | 0.216 | 0.851 | 0.030 |
| Jailbreak classification | NeuralTrust/prompt-guard-oss-small | 5888 | 0.235 | 0.681 | 0.349 | 0.268 | 0.319 |

Presidio uses default English recognizers and spaCy `en_core_web_lg` 3.8.0; relevant PERSON/identifier entity types were requested. Exact TAB `DIRECT PERSON/CODE` is a narrower target than Presidio's detection taxonomy. No baseline threshold was tuned on this test set.

Meta Prompt Guard 86M is manual-gated and was not accessible. ProtectAI DeBERTa v2's model card says it does not detect jailbreak attacks, so comparing it on the jailbreak benchmark would conflate tasks. Secret scanners were not run because authorized real-secret gold was unavailable. No baseline values were imputed.

## BERT training-source contamination

The BERT model card identifies `jackhhao/jailbreak-classification` as training data, whose card cites the same jailbreak prompt repository as this evaluation. We screened both published train CSVs at revision `2f2ceeb39658696fd3f462403562b6eea5306287` (2642 rows). Of 5888 evaluated prompts, 616 exact and 64 near training matches were removed for the shared clean-row comparison. Even the remaining rows share a source family and are **not a source-independent test of BERT**.

| System | Clean N | Precision | Recall | F1 | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| TrustLaya-S | 5208 | 0.003 | 0.923 | 0.005 | 0.899 | 0.077 |
| BERT | 5208 | 0.003 | 0.923 | 0.005 | 0.849 | 0.077 |

Do not use the unfiltered BERT row above for a model ranking. The clean-row comparison is a contamination diagnostic and still has source-family leakage risk.

## NeuralTrust provenance limit

`NeuralTrust/prompt-guard-oss-small` is MIT licensed and explicitly classifies direct jailbreaks; revision `40e5c56b68a1b081484c3e56c9bee726e2749138` was run on the same retained prompts. Its model card says it was fine-tuned on a **private** dataset, so training overlap cannot be verified. Its score is a compatible-task reference with **unknown contamination status**, not an independent clean baseline or leaderboard claim. It reads up to 512 tokens while TrustLaya-S reads 96.
