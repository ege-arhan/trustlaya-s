# Independent Council review: evidence audit

The supplied Council note had **2 of 8 members respond**, plus a chair synthesis. We treated it as a partial critique and checked its claims against frozen artifacts and current code. It does not itself supply human labels or hardware measurements.

| Claim or recommendation | Finding | Action |
|---|---|---|
| 460 reviews are underpowered | Valid concern. Counts per intent/vector were unknown and 460 could undersample rare cases. | Expanded local packet to **600**, stratified by source, three length groups and frozen-model disagreement. This does **not** guarantee per-class sample size. |
| Eight mutually exclusive fine labels conflate intent, vector and quotation | Valid. | Replaced the review form with hierarchical Level 1 intent, conditional Level 2 vector/subtype, and independent PII/secret/obfuscation flags. `UNRESOLVED` routes to adjudication and never becomes gold. |
| Two reviewers need a senior adjudicator and alpha | Valid for conflict resolution. | Added third distinct ID, full-record conflict handling, and Krippendorff nominal alpha for Level 1 intent. No alpha is claimed before real reviews. The software cannot verify human identities. |
| Zero human gold makes every old F1 invalid | **Overstated.** Historical F1 values are measured against existing dataset/source labels and remain valid *for those proxies*. They do not establish independently reviewed attack intent or universal performance. | Retained frozen metrics with explicit task/provenance limits. |
| PII and injection share one head | **Incorrect.** Existing v3 PII token head and attack sequence head are separate; the v2 shared encoder has separate risk logits. | No architecture change. |
| Threshold was tuned on the final JLL test | **Not supported for V4.** `select_v4_operating_point.py` fits calibration and threshold on DEV halves; `evaluate_v4_final.py` reads the frozen operating point. JLL had been examined in prior v2/v3 work, so it is not a fresh blind source. | Kept this caveat; no new threshold tuning. |
| 94-token bottleneck | Measured: 612/1,168 new candidates and 94.5% of historical JLL positives exceed 94 tokens. | Ran a separate no-training 94 vs native-510/head-tail/sliding ablation on the old proxy DEV. Results support partial recall recovery for V4, with higher FPR. |
| Turkish tokenizer fragments English text | Supported **as a token-length difference**, not proven as a cause of model errors. | Compared to a pinned multilingual tokenizer; see [tokenizer audit](v5_tokenizer_audit.md). No tokenizer swap under frozen weights. |
| UNO Q >500 ms / <150 ms or ECE ≤0.08, FAR ≤0.001 as facts | Unsupported: no physical UNO Q run or validated GOLD exists. These are proposed gates, not measurements. | No hardware or production claims. |
| MinHash, language and source-separated hidden tests | Material unfinished work. Existing exact and char n-gram cosine checks found one prior DEV near match and excluded it from the packet. | MinHash/semantic leakage screen and two untouched hidden sources remain prerequisites before headline evaluation. |

**Decision:** no V5 training, publication, or gateway switch. The 600-row packet remains a candidate set with zero gold labels. The first real gate is independent human review with adjudication and enough attack/benign coverage per source. Council numerical targets are research aspirations until validated on source-separated data and the actual device.
