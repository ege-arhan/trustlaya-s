# V4 isolated training record

V2 and v3 were frozen before V4 work; checksums and exact row predictions are in [the frozen comparison](v2_v3_frozen_comparison.md). V4 only adds a v4-linear-head-1 linear attack-intent head on the same frozen encoder. The head has **513 parameters** (512 weights and one bias), stored in a 2,188-byte safetensors file. Head SHA-256: `fa72302753553b0accaedb0feb92c02f0313c31f13debc2903dd408873788a1c`. V2 weight SHA-256: `99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c`. Tokenizer SHA-256: `d1982e608b0883c5840b3dc9a735af3fbc45f3b75417dea44fa3efab3854346f`. The new head is local at `models/trustlaya-s-v4-research/attack_intent_head.safetensors` and is not wired into inference defaults or the gateway.

TRAIN: 2173 rows, 95 composed JailbreakChat attacks across 89 surviving template-prefix groups and 2078 WildChat negatives. Encoder max input 96 total tokens for head fitting, mean pooling over non-padding encoder outputs. Optimizer: scikit-learn logistic regression with C=0.1, balanced class weights, max_iter=500, seed=20260925. No encoder fine-tuning, no v2/v3 checkpoint edits. The 95 positive rows are **small and weakly labeled**, so this is an exploratory head, not a robust human jailbreak model.

Read strategy chosen on source-separated proxy DEV: WINDOW_MAX over 94-token overlapping content windows (47-token overlap). DEV was split by stable text hash: 230 calibration rows and 215 threshold-selection rows. Logistic score calibration fitted only on calibration rows. Selection fixed threshold 0.79 by maximizing F1 under FPR <=0.25, with recall >=0.80 preferred where feasible; that target was not feasible. Final JLL and deepset test data did not fit the head, calibration or threshold.

Source checkout setup after the [v2/v3 prerequisite](../README.md#external-data-v3-experiment-not-deployed):

```bash
git clone https://github.com/wagner-group/JailbreaksOverTime.git benchmarks/external/raw/v4/jailbreaks_over_time
git -C benchmarks/external/raw/v4/jailbreaks_over_time checkout 94a2e998282301d545f92177e3fff8aab11fb0dd
git clone https://github.com/GenAI-Security-Project/GenAI-LLM-Top10.git benchmarks/external/raw/v4/owasp
git -C benchmarks/external/raw/v4/owasp checkout 9253e38ade58e959b531c0c5c9a4842272c9cd0e
```

Run in order: `PYTHONPATH=src .venv/bin/python scripts/freeze_v4_baselines.py`; `scripts/v4_diagnose_context.py`; `scripts/prepare_v4_dev.py`; `scripts/prepare_v4_train.py`; `scripts/check_v4_leakage.py`; `scripts/train_v4_head.py`; `scripts/evaluate_v4_windows.py --head v3`; `scripts/evaluate_v4_windows.py --head v4`; `scripts/select_v4_operating_point.py`; `scripts/evaluate_v4_final.py`; `scripts/evaluate_v4_unseen_source.py`; `scripts/benchmark_v4_latency.py`; `scripts/analyze_v4_errors.py`; `scripts/plot_v4_reliability.py`; `scripts/write_v4_reports.py`. The Bordair and deepset files are downloaded at pinned revisions by the preparation/evaluation scripts. Requires the ignored frozen v2/v3 checkpoints and the repository virtual environment. No checkpoint is published.
