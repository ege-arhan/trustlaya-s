# Evaluation

`reports/evaluation.json` measures accuracy, precision, recall, F1, false positive rate, false negative rate and support for each task on the held-out synthetic test. `reports/calibration.json` compares raw and temperature-scaled ECE, Brier and NLL on the synthetic validation set. `reports/teacher_baseline.json` uses a separate 128-row test subset. `reports/quantization.json` compares FP32 ONNX and INT8 on 256 test rows. `benchmarks/edge.json` records one-text batch-1 cold start and warm latency on this MacBook.

Synthetic templates make these results optimistic for PII and secret patterns. Domain transfer, adversarial paraphrases, unseen PII formats and real-world false positives remain unmeasured. INT8 action disagreements are reported and require review before deployment.
