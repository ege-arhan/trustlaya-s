# Evaluation

`reports/evaluation.json` measures accuracy, precision, recall, F1, false positive rate, false negative rate and support for each task on the held-out synthetic test. `reports/calibration.json` compares raw and temperature-scaled ECE, Brier and NLL on the synthetic validation set. `reports/teacher_baseline.json` uses a separate 128-row test subset. `reports/quantization.json` compares FP32 ONNX and INT8 on 256 test rows. `benchmarks/edge.json` records one-text batch-1 cold start and warm latency on this MacBook.

Synthetic templates make these results optimistic for PII and secret patterns. Domain transfer, adversarial paraphrases, unseen PII formats and real-world false positives remain unmeasured. INT8 action disagreements are reported and require review before deployment.

## Advanced candidate

The advanced reports have distinct scopes. `reports/advanced_evaluation.json` evaluates all 1,975 synthetic test rows: mean task accuracy 0.9066, macro F1 0.6632, security F1 0.7769, PII F1 1.0000, injection F1 0.5549. This test split is mixed-language only and has 143 normalized prototypes across the complete 10,000-row dataset; the perfect PII number should not be generalized.

`reports/advanced_pii_experiment.json` measures scenario-disjoint development and test rows from a separate MIT-licensed Turkish privacy source. The v2 head improved PII F1 on its 979-row test while increasing false positives. `reports/independent_pii_v2.json` evaluates a different CC-BY synthetic Turkish PII source, without training or threshold selection on it: v1 F1 0.737 and v2 F1 0.784 at the development-selected threshold; v2 false-positive rate remains 0.316. This source was subsequently inspected, so it is no longer fresh for another selection round. No benchmark result in this repository is a production field estimate.

`reports/tr_benchmark.json` is a 33-case researcher-authored language smoke test. The 11 cases per TR/EN/mixed slice give large sampling uncertainty and cannot establish multilingual robustness. `reports/adversarial.json` has 24 related families and a complete encoding failure. `reports/onnx_v2_parity.json` compares FP32 and INT8 on 256 synthetic rows; INT8 disagrees with FP32 policy action on 5.1%. `benchmarks/results.json` has measured batch-1 latency and size. See [calibration](calibration.md), [adversarial evaluation](adversarial.md), and [dataset card](../DATA_CARD.md).

### Common-scope PII baseline

On the same 2,000-row BTX24 task mapping, the rule-only detector achieved F1 **0.480**, recall **0.316**, and FPR **0.000**. V1 hybrid achieved F1 **0.737**, recall **0.861**, FPR **0.474**; v2 hybrid achieved F1 **0.784**, recall **0.848**, FPR **0.316** at its separately selected 0.8 threshold. The rules trade recall for precision. These comparisons share cases and task definition, but the source itself is synthetic and has been inspected. Exact confusion matrix and calibration values are in `reports/independent_pii_v2.json`.

### Isolated injection candidates

The bilingual BPI-only and joint heads are local research candidates, not the released v2. Development-selected joint training reached F1 **0.869** on BPI's transformed test and **0.868** on a new PolyGuardBench cross-axis check, versus released v2 **0.561** and **0.486** respectively. It regressed on the original mixed-only synthetic test (**0.555→0.464**) and flagged **80%** of benign AgentInjectionBench tool returns. This is a mixed result, not a safe overall score increase. Source scopes, overlap exclusions, all false-positive rates, and exact counts: `reports/injection_experiments.md`.
