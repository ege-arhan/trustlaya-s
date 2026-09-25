# V5 calibration status

No human-reviewed DEV/calibration split or V5 model exists. No V5 ECE, Brier, NLL, reliability diagram, or selected threshold is reported. `benchmarks/v5/calibration.json` and `thresholds.json` explicitly mark this state.

The old proxy strategy run reused frozen v3/v4 Platt fits after aggregation. Those fits were not designed for the new pooling rules. The machine file exposes diagnostic ECE/Brier only. The prior V4 frozen JLL test showed calibration transfer failure (ECE 0.665, Brier 0.556 at its selected calibration), so a lower proxy F1/FPR cannot justify confidence claims. Once human-reviewed DEV exists, reserve distinct calibration and selection partitions, fit each strategy only on calibration data, freeze before hidden tests, and report raw/calibrated ECE, Brier, NLL and reliability curves.
