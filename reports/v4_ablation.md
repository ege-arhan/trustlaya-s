# V4 controlled comparisons

V2, v3 and v4 on the **same frozen 5,761-row JLL cohort** (fixed thresholds):

| Model | Training change | Read | F1 | Recall | FPR |
|---|---|---|---:|---:|---:|
| v2 | frozen baseline | HEAD 94 | 0.206 | 0.946 | 0.898 |
| v3 | frozen Gandalf/prompts.chat attack head | HEAD 94 | 0.048 | 0.035 | 0.052 |
| v4 | frozen encoder + JOT-fitted linear head | WINDOW_MAX | 0.250 | 0.901 | 0.657 |

The v4-to-v3 comparison changes **both** head fit and read strategy, so the test difference cannot be attributed to one factor. On source-separated proxy DEV at raw threshold 0.50, v4 HEAD F1=0.204 and WINDOW_MAX F1=0.422; this supports a window effect on that DEV only. No separate v4 JLL HEAD result, no without-hard-negative fit and no causal loss-function ablation were run. The 95 positive training examples remaining after leakage screening make extra fitted variants particularly unstable. Calibration and threshold selection used DEV only. “False allow” and “false block” are not claimed: this head was never connected to a policy decision or guarded tool execution path.
