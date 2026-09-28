# V4 calibration transfer

Calibration fitted on 230 Bordair/OWASP DEV rows, threshold selected on another 215 rows. The selection half's ECE was 0.458 raw versus 0.127 after calibration; Brier 0.366 versus 0.180; NLL 1.329 versus 0.543. These are descriptive for a game-attack/documentation mixture with 161/215 positives, not population risk probabilities.

On frozen JailbreakLLMs test (635/5,761 positives), **calibration did not transfer**: ECE 0.263 raw versus 0.665 calibrated; Brier 0.191 versus 0.556; NLL 0.566 versus 1.486. The [reliability diagram](v4_reliability.svg) visualizes this shift. Scores must not be read as deployment risk probabilities. A future calibration cohort must match the intended operational prompt distribution and remain separate from training and test.
