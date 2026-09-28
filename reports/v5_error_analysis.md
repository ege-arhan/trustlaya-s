# V5 error-analysis status

No V5 model and no human-reviewed benchmark labels exist, so there are **no valid V5 false positives or false negatives to inspect**. Existing v4 errors remain in [its frozen analysis](v4_error_analysis.md). Do not relabel historical errors as V5 results.

After a strategy, threshold and calibration are frozen on reviewed DEV, inspect at least 100 false positives and 100 false negatives when available on the held-out sources. Human reviewers will record educational discussion, quotation, documentation, benign security vocabulary, long benign text, and Turkish/mixed language for FPs; and attack beyond first window, long/end attacks, indirect instruction, paraphrase, obfuscation, role hijack and prompt extraction for FNs. Quantify counts and multiple applicable categories. Hidden test examples must not be opened for model/threshold selection.
