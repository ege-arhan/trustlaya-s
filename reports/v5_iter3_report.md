# V5 iteration 3: short human attacks (HackAPrompt)

**V5_STATUS = REGRESSION** under the pre-registered rules ([iter3_plan.md](iter3_plan.md)): the
three-seed mean deepset recall is 0.299, below the 0.50 floor. The A4 control (no HackAPrompt) is
also below that floor (0.382), so the label reflects deepset recall staying low rather than a new
drop caused by HackAPrompt. **V2 stays the default and the firewall model; no V5 weights are
published.** Thresholds and temperatures were frozen in [`v5_iter3_lock.json`](v5_iter3_lock.json)
(class-balanced DEV rule, declared in the plan) before one TEST run. deepset, Gandalf and MASSIVE
test were scored in earlier iterations (reused, not blind); the MASSIVE dev partition was scored
here for the first time. Master: [`v5_iter3_master.json`](v5_iter3_master.json).

## Config means over three seeds (TEST, each run at its own frozen DEV threshold)

| Metric | A4 control (no HackAPrompt) | H1 = A4 + HackAPrompt |
|---|---|---|
| Tensor Trust recall | 0.993 ± 0.002 | 0.988 ± 0.003 |
| JailbreakLLMs recall | 0.882 ± 0.003 | 0.828 ± 0.026 |
| JailbreakLLMs FPR | 0.266 ± 0.031 | 0.189 ± 0.024 |
| HackAPrompt recall | 0.861 ± 0.023 | 0.999 ± 0.001 |
| HackAPrompt recall, no 'pwn' | 0.843 ± 0.023 | 0.997 ± 0.003 |
| deepset recall | 0.382 ± 0.052 | 0.299 ± 0.039 |
| deepset FPR | 0.157 ± 0.030 | 0.070 ± 0.013 |
| deepset PR-AUC | 0.626 ± 0.015 | 0.670 ± 0.018 |
| Gandalf recall | 0.985 ± 0.003 | 0.978 ± 0.008 |
| MASSIVE test FPR (reused) | 0.463 ± 0.130 | 0.262 ± 0.043 |
| **MASSIVE dev FPR (fresh)** | 0.441 ± 0.127 | 0.247 ± 0.044 |
| security docs FPR | 0.037 ± 0.006 | 0.010 ± 0.010 |
| arXiv FPR | 0.022 ± 0.000 | 0.015 ± 0.006 |
| TEST1 ECE raw | 0.029 ± 0.006 | 0.049 ± 0.002 |
| TEST1 ECE calibrated | 0.027 ± 0.006 | 0.017 ± 0.003 |
| MASSIVE dev FPR en / de / tr | 0.488 / 0.434 / 0.401 | 0.339 / 0.175 / 0.228 |

Rules: deepset ✗, fresh MASSIVE ✗ (0.247 > 0.20; English 0.339 > 0.30), Tensor Trust ✓,
JailbreakLLMs ✓, HackAPrompt ✓ (also without "pwn"), docs/arXiv ✓, calibration ✓.

## Runs

| Run | DEV threshold | Temperature | Checkpoint | deepset recall | deepset FPR | MASSIVE dev FPR | HackAPrompt recall (no pwn) |
|---|---|---|---|---|---|---|---|
| H1_grouped_s42 | 0.37 | 0.6617 | `617c678ad82bc306…` | 0.254 | 0.056 | 0.202 | 0.997 |
| H1_grouped_s43 | 0.24 | 0.6717 | `d89341018065cc73…` | 0.325 | 0.081 | 0.291 | 1.000 |
| H1_grouped_s44 | 0.3 | 0.6717 | `7b054caaf3be2755…` | 0.318 | 0.073 | 0.248 | 0.993 |
| H1_random_s42 | 0.32 | 0.6717 | `d9c5e2f97c8e5555…` | 0.270 | 0.056 | 0.237 | 1.000 |
| A4control_s42 | 0.15 | 0.8541 | `69995a93efbe9a20…` | 0.436 | 0.192 | 0.585 | 0.869 |
| A4control_s43 | 0.2 | 0.907 | `f9524a672b2a6248…` | 0.333 | 0.139 | 0.348 | 0.835 |
| A4control_s44 | 0.23 | 0.9347 | `1cb28ce012dd5ef1…` | 0.377 | 0.141 | 0.389 | 0.825 |

H1_random_s42 (E6-style random batches) sits inside the H1 grouped range, consistent with the
variance study: batching is not a material factor.

## What changed and what did not

- HackAPrompt is learned fully in-source (recall ≥ 0.99 with or without the target word "pwn"),
  so the "PWNED" shortcut is not the whole story there, and false alarms fall almost everywhere
  (deepset FPR 0.157 → 0.070, fresh MASSIVE FPR 0.441 → 0.247, JailbreakLLMs FPR 0.266 → 0.189).
- deepset attack ranking improves only a little (PR-AUC 0.626 → 0.671) and recall at the DEV
  threshold does not (0.382 → 0.299).
- Diagnostic on deepset attacks (analysis only, no selection): about one third are German (83 of
  252) and 100 end with a question mark. H1 catches 10% of German deepset attacks and 29% of English
  ones. Every attack source in TRAIN is English; the model has seen German benign text but no German
  attacks.

## Next steps (not run)

1. Human-written non-English attacks are the missing piece. No permissively licensed human German
   or Turkish injection set was found so far; machine-translated attacks would be synthetic and
   must be labeled as such if used.
2. Question-shaped injections ("Was ist X? Ignoriere ... und ...") are under-represented in every
   attack source; a source with that form, independent of deepset, is needed.
3. Keep V2 as default; do not publish these checkpoints.
