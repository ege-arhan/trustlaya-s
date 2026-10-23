# TrustLaya-S V4 research verdict: NO-GO

## Question and measured answer

Could a frozen 42M-parameter encoder, a small externally fitted attack head and long-context reading recover v3's missed direct jailbreaks **without** v2-level false alarms? **Partly on JLL, but not reliably across sources.** Paired JLL v4 recall=0.901, F1=0.250, FPR=0.657; v2 recall=0.946, F1=0.206, FPR=0.898. Independent deepset prompt-injection test v4 recall=0.067, F1=0.125 versus v2 F1=0.765. Deepset labels cover a related but distinct task and are reported separately.

## What changed

V2/v3 evaluation artifacts were frozen with checksums; no baseline checkpoint or default gateway was changed. Actual v2/v3 read length is 94 content tokens; 94.5% of JLL attacks exceed that. Six read strategies were measured before fitting. A V4 research-only 513-parameter linear head was fitted on 95 composed JailbreakChat positives and 2078 real WildChat negatives after source/near-duplicate screening. WINDOW_MAX and a DEV-only threshold of 0.79 were fixed before frozen final evaluation. Source roles, labels, tokenizer corrections, and exact scripts are in [data provenance](v4_data_provenance.md), [label audit](v4_label_audit.md), and [training](v4_training.md). Final verification: **83/83 pytest tests passed**, frozen baseline checksums verified, and all 11 V4 JSON reports parsed successfully.

## Scientific limits and deployment decision

V4's JLL false alarm rate 0.657 is still far too high for default blocking, and increases to 0.860 above 510 tokens. DEV calibration degraded on JLL (ECE 0.665, Brier 0.556). The nominal train attack corpus has only 89 non-overlapping template groups and uses composed examples; its benign labels are not individually human-reviewed. DEV positives are game attacks, DEV negatives are explanatory documents. There is no independent Turkish test and no fully human-reviewed six-way intent taxonomy. The JLL final source had been inspected in prior v2/v3 work, though V4 never trained or selected on it. No UNO Q measurement was made. V4 is **NO-GO**, remains local and disconnected from the default gateway; v2 remains the frozen baseline.

## Next priority

Acquire a licensed, independently annotated **direct-jailbreak** corpus with genuinely human-written long prompts and a matched human benign-security corpus. Review intent labels and quoted attacks by humans, reserve a never-inspected source, then test a window-count-aware aggregation rule with a false-alarm constraint. Keep policy and tool authorization fail-closed regardless of classifier error.

See [scorecard](v4_external_scorecard.md), [long context](v4_long_context.md), [ablation](v4_ablation.md), [calibration](v4_calibration.md), and [error analysis](v4_error_analysis.md) for measured evidence.
