# Limitations

Risk score is not a legal or ethical verdict. Probabilities are task-model outputs and require task-specific calibration.

The dataset is generated from a small number of controlled templates, so held-out template families do not simulate open-world distribution shift. The Turkish-first encoder has limited English pretraining; mixed-language performance was not separately established. Teacher output is often wrong on these specialized tasks and is weakly weighted. Confidence is categorical decision sharpness, not calibrated probability of correctness. Severity and action heads are trained on generated labels. PII/secret evidence rules cover listed formats only and may miss variants or create false positives in real data. Agent permission risk is policy-only. Arduino UNO Q deployment is documented but untested. INT8 quantization changes some final actions and is experimental.
