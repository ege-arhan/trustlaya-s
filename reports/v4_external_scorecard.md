# V4 external scorecard — NO-GO

All metrics below are measured with a **fixed before-test** operating point: v2 raw threshold 0.50; v4 WINDOW_MAX plus DEV-fitted calibration at 0.79. Deepset's prompt-injection label is **not the same target** as JailbreakLLMs direct-jailbreak intent. A high score on one must not be pooled with the other. V4 is research only.

| Dataset | Provenance | Split | N | Language | v2 Precision | v2 Recall | v2 F1 | v2 FPR | v4 Precision | v4 Recall | v4 F1 | v4 FPR |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JailbreakLLMs | Reddit/Discord/website community | frozen clean test | 5,761 | not annotated | 0.115 | 0.946 | 0.206 | 0.898 | 0.145 | 0.901 | 0.250 | 0.657 |
| deepset prompt-injections | deepset public dataset | official test | 116 | not annotated | 0.684 | 0.867 | 0.765 | 0.429 | 1.000 | 0.067 | 0.125 | 0.000 |

Paired JLL confusion matrices: v2 TP=601, FP=4603, FN=34; v3 TP=22, FP=267, FN=613; v4 TP=572, FP=3370, FN=63. V3 JLL F1=0.048, recall=0.035, FPR=0.052. No V4 PII claim: the V2/V3 frozen TAB PII track and v3 exact-span F1 remain unchanged.

Unseen deepset official test: v3 F1=0.298, recall=0.283, FPR=0.661; v4 missed 56 of 60 positives. Label provenance is sparse and the task differs, but this is clear adverse transfer evidence. No verified Turkish external score. [Exact machine-readable JLL metrics](v4_final_external_metrics.json); [deepset metrics](v4_unseen_external_metrics.json); raw text-free row predictions are local/gitignored.
