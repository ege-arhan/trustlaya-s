# Native-context ablation before V5 training

The frozen v2 encoder has **512 total position embeddings**, leaving **510 content tokens** after `[CLS]`/`[SEP]`. Its normal application request is 96 total / 94 content tokens. The [machine-readable ablation](../benchmarks/v5/old_proxy_native_context_metrics.json) uses the **old V4 proxy DEV** of 340 Bordair game attempts and 105 OWASP explanatory paragraphs. These are source-derived labels, not human-reviewed intent gold. The original v2/v3/v4 heads, fits and thresholds remain fixed; no checkpoint or gateway changed.

| V4 reading | Recall | FPR | F1 | Mean windows/input | Mac MPS P50/P95 ms* |
|---|---:|---:|---:|---:|---:|
| First 94 content tokens | 0.432 | 0.181 | 0.581 | 1.00 | 4.85 / 5.73 |
| Native first 510 | 0.626 | 0.210 | 0.741 | 1.00 | 6.51 / 47.68 |
| First 255 + last 255 | 0.621 | 0.210 | 0.736 | 1.00 | 6.32 / 9.91 |
| Sliding 256, stride 128, max score | 0.679 | 0.210 | 0.779 | 1.62 | 6.40 / 17.59 |

\*Latency uses 40 deterministic length-spanning inputs, jointly scoring v2/v3/v4 heads after one shared encoder pass. It is a Mac measurement, **not UNO Q**. Native 510 P95 includes a measured long-input outlier. Network and policy are excluded.

V3 recall stayed at 0.003–0.006 at its fixed 0.30 threshold, so longer reading **did not repair v3** on this proxy. V2 recall stayed around 0.96 but FPR remained 0.971. V4's longer reading recovered source-positive detections, with a higher false alarm rate than first-94 reading. Thus truncation contributes to exposure loss, but the resulting score shift and source/class confounding prevent attribution of the full failure to truncation alone. Fits calibrated on 94-token/window-max outputs were not refit; these numbers are exploratory, not an operating point.

The Council's suggested synthetic head/middle/tail shift can prove a mechanical pipeline property, but synthetic placement is not independent real-world performance evidence. Gold attack locations and error slices still require human review. The 510-token comparison must be repeated on source-separated, human-reviewed DEV before strategy selection; hidden tests remain sealed.
