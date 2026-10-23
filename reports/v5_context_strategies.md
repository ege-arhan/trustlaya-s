# Eight context strategies: pre-V5 exploratory measurement

**Do not use this table for V5 selection.** It reuses V4's old development proxy: 340 Bordair game attempts and 105 OWASP explanatory paragraphs (N=445, 76.4% source-positive). The per-row labels are inherited from source context, **not two-human intent review**. V4's head, calibration, and threshold 0.79 were already selected using this source family. All eight strategies were run without training or threshold adjustment. The [full machine results](../benchmarks/v5/old_proxy_context_metrics.json) include v2/v3/v4, all metrics, fixed thresholds, and the 40-input latency sample; raw text and row predictions remain local.

| V4 strategy | Precision | Recall | F1 | FPR | FNR | Mean windows/input | MPS P50/P95 ms* |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIRST_WINDOW | 0.886 | 0.432 | 0.581 | 0.181 | 0.568 | 1.00 | 5.66 / 7.59 |
| LAST_WINDOW | 0.913 | 0.341 | 0.497 | 0.105 | 0.659 | 1.00 | 5.66 / 7.04 |
| HEAD_TAIL | 0.920 | 0.406 | 0.563 | 0.114 | 0.594 | 1.00 | 5.62 / 7.11 |
| FIXED_SLIDING_WINDOW (mean logits) | 0.902 | 0.326 | 0.479 | 0.114 | 0.674 | 5.20 | 8.55 / 75.16 |
| MAX_WINDOW_SCORE | 0.898 | 0.703 | 0.789 | 0.257 | 0.297 | 5.20 | 8.52 / 21.56 |
| MEAN_WINDOW_SCORE | 0.911 | 0.424 | 0.578 | 0.133 | 0.576 | 5.20 | 8.45 / 22.08 |
| TOP_K_RISK_WINDOW (top two mean) | 0.914 | 0.624 | 0.741 | 0.190 | 0.376 | 5.20 | 8.43 / 21.31 |
| TWO_STAGE_CONTEXT_ROUTER | 0.894 | 0.568 | 0.694 | 0.219 | 0.432 | 2.12 | 6.58 / 8.86 |

\*Measured on 40 deterministic length-spanning inputs on this Mac's MPS. Latency includes token-window preparation, encoder and all three existing heads but excludes network, gateway policy and UNO Q. Strategy latency is shared across heads; it is **not** an independent v4-only latency measurement. The large fixed-sliding P95 includes a measured outlier and is retained. `mean windows/input` counts expensive semantic windows, not batched `forward()` invocations.

The router always samples the first and last window and at most two additional windows around a short, fixed list of instruction/role cues. The cues only retrieve regions for semantic inspection; they do not decide attack truth. It reduced mean windows from 5.20 to 2.12 on this proxy, while also losing recall relative to max pooling. V2 remained near 0.95 recall **with 0.95–0.98 FPR** across strategies; V3 remained near-zero recall. These proxy numbers show no validated V5 gain.

The existing v3/v4 Platt fits were applied after each new aggregation for diagnostic comparability. They were **not** refit per strategy, so ECE/Brier in the machine file are diagnostic, not calibrated production probabilities. Future human-reviewed DEV must provide separate calibration and threshold selection; hidden tests must remain unopened until then.
