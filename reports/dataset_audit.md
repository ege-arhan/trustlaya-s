# Synthetic dataset audit

Audited the unchanged 10,000-row local splits. Source: controlled templates only; no independently adjudicated labels.

Exact unique texts: 2,682; normalized unique prototypes: 143. Exact repeated rows: 7,318; normalized repeated rows: 9,857.

| Split | Rows | Families | Turkish | English | Mixed | Exact repeated | Normalized repeated |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6972 | 77 | 3943 | 3029 | 0 | 5066 | 6872 |
| val | 1053 | 11 | 0 | 1053 | 0 | 856 | 1038 |
| test | 1975 | 22 | 0 | 0 | 1975 | 1396 | 1947 |

**Critical split confound:** all original test rows are labeled mixed-language; validation is English-only; training contains Turkish and English but no mixed rows. Thus original held-out scores cannot be reported as separate Turkish or English test performance.

| Split pair | Exact text overlap | Normalized overlap | Family overlap |
|---|---:|---:|---:|
| train-val | 0 | 0 | 0 |
| train-test | 0 | 0 | 0 |
| val-test | 0 | 0 | 0 |

Near-duplicate proxy using character TF-IDF cosine on normalized templates:

| Split | Rows cosine ≥ 0.80 to another split | Rows cosine ≥ 0.90 | Rows token Jaccard ≥ 0.80 |
|---|---:|---:|---:|
| train | 0 | 0 | 0 |
| val | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

These similarity counts flag possible shared phrasing; they are not manual paraphrase labels. Family-disjoint splitting removes identical template families but cannot remove semantic overlap among different templates. Repeated literals and small template pool create strong memorization risk.

## Label distribution

| Task | Train positive | Validation positive | Test positive | Total positive |
|---|---:|---:|---:|---:|
| pii | 1270 | 192 | 356 | 1818 |
| secret | 798 | 88 | 201 | 1087 |
| prompt_injection | 714 | 93 | 279 | 1086 |
| dangerous_instruction | 602 | 111 | 196 | 909 |
| privacy_risk | 1300 | 197 | 321 | 1818 |
| security_risk | 3738 | 581 | 1135 | 5454 |
| ethics_risk | 1265 | 212 | 341 | 1818 |
| oversight_risk | 1265 | 184 | 369 | 1818 |
| data_governance_risk | 641 | 97 | 171 | 909 |

Category, language and all template-family counts are in `reports/dataset_audit.json`. The new Generalization and TR benchmarks should allocate held-out families separately within each language and independently label external cases.
