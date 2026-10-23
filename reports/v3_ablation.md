# V3 controlled ablation and promotion decision

V2 and v3 rows are paired after a predeclared char-ngram near-duplicate filter. The v2 full benchmark remains frozen separately. The v3 encoder and all v2 task heads were held fixed. Added trainable parameters: 2,565 for a five-tag PII head and 513 for a binary attack head. They were trained on distinct external data. No architecture-size increase or uncontrolled search occurred.

| Configuration | TAB window F1 / recall / FPR | Jailbreak F1 / recall / FPR | Interpretation |
|---|---|---|---|
| A. Frozen v2, original 0.50 raw threshold | 0.135 / 0.165 / 0.077 | 0.206 / 0.946 / 0.898 | Paired clean baseline |
| B. Frozen encoder + TAB train token head | 0.158 / 0.107 / 0.015 | — | Better ranking/FPR, lower recall |
| C. B + development-selected BIO decoding and code-span cleanup | Exact span F1 0.079 | — | Exact span remains weak; PERSON F1 0.039 |
| D. Frozen encoder + Gandalf positives + prompts.chat negatives | — | 0.048 / 0.035 / 0.052 | Dramatic recall collapse on new source |
| E. D + development Platt calibration/threshold | — | Same operating point; Brier 0.141→0.126, ECE 0.119→0.121 | Calibration transfer fails on ECE/NLL |
| F. Combined experimental heads | Same as B/C | Same as D/E | **NO-GO** |

The linear PII token head caused the measured TAB ranking/FPR change. Exact-span code cleanup improved exact entity offsets on development data, but it cannot fix person-name recall. The additional human benign data reduced jailbreak false alarms, yet the positive game-prompt source did not cover the direct community jailbreak patterns. This is a genuine generalization failure, not a threshold-selection success.

As a within-source diagnostic, the v3 attack head detected 111/112 Gandalf official test attacks, while it detected only 22/635 independent JailbreakLLMs attacks. The Gandalf split has no negative examples and does not support F1 or false-alarm estimates.

The requested “v3 with external-data fine-tuning” here means head-only training on external data; the shared encoder was **not** fine-tuned. A separate encoder fine-tune and a v3 without the token head were not run, so their marginal effects are unmeasured. No claim is made for either. The v2 deployed checkpoint, calibration, ONNX and policy are unchanged.
