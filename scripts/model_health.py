"""Aggregate measured health signals without inventing missing metrics."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    evaluation = read("reports/advanced_evaluation.json")
    calibration = read("reports/calibration_v2.json")
    adversarial = read("reports/adversarial.json")
    benchmark = read("benchmarks/results.json")
    parity = read("reports/onnx_v2_parity.json")
    external = read("reports/independent_pii_v2.json")
    agent_path = ROOT / "reports/agent_injection_independent.json"
    agent = json.loads(agent_path.read_text()) if agent_path.exists() else None
    v2_onnx = next(row for row in benchmark if row["model"].startswith("TrustLaya-S v2")
                   and row["backend"] == "ONNX CPU FP32")
    worst_variant = min((key for key in adversarial["results"]
                         if key not in ("all", "clean", "adversarial", "tr", "en", "mixed")),
                        key=lambda key: adversarial["results"][key]["f1"])
    report = {
        "model": "TrustLaya-S v2 PII candidate",
        "test": {"scope": evaluation["dataset"],
                 "macro_f1": evaluation["macro_f1"],
                 "mean_task_accuracy": evaluation["mean_task_accuracy"],
                 "security_f1": evaluation["tasks"]["security_risk"]["f1"],
                 "pii_f1": evaluation["tasks"]["pii"]["f1"],
                 "injection_f1": evaluation["tasks"]["prompt_injection"]["f1"]},
        "calibration": {"scope": calibration["scope"],
                        "raw_mean": calibration["mean"]["raw"],
                        "calibrated_mean": calibration["mean"]["calibrated"]},
        "adversarial": {"scope": adversarial["dataset"],
                         "clean_f1": adversarial["results"]["clean"]["f1"],
                         "transformed_f1": adversarial["results"]["adversarial"]["f1"],
                         "worst_variant": worst_variant,
                         "worst_variant_f1": adversarial["results"][worst_variant]["f1"]},
        "independent_pii": {"scope": external["scope"],
                            "rule_f1": external["results"]["rule_only"]["f1"],
                            "v1_f1": external["results"]["v1"]["dev_tuned"]["hybrid_f1"],
                            "v2_f1": external["results"]["v2"]["dev_tuned"]["hybrid_f1"],
                            "v2_false_positive_rate": external["results"]["v2"]["dev_tuned"]["false_positive_rate"]},
        "edge": {"size_mib": v2_onnx["size_mib"],
                 "onnx_cpu_p50_ms": v2_onnx["warm_p50_ms"],
                 "int8_final_action_disagreement": parity["drift"]["int8"]["final_action_disagreement_rate"]},
        "uncertainty": {"confidence_distribution": evaluation["confidence_distribution"],
                        "abstention_rate": evaluation["abstention_rate"],
                        "coverage_risk": evaluation["selective_prediction"],
                        "note": "High categorical confidence does not predict lower synthetic action error in this run; do not treat it as correctness probability."},
    }
    if agent:
        report["independent_agent_injection"] = {
            "scope": agent["scope"],
            "single_window": agent["views"]["tool_results"]["at_0_5"],
            "sliding_max": agent["views"]["tool_results_sliding_max"]["at_0_5"]}
    (ROOT / "reports/model_health.json").write_text(json.dumps(report, indent=2))
    lines = ["# TrustLaya-S v2 model health", "", "Research diagnostics only. All performance figures below have their own dataset scope; do not combine them into a production claim.", "", f"Synthetic mixed-only test: macro F1 **{report['test']['macro_f1']:.3f}**, security F1 **{report['test']['security_f1']:.3f}**, PII F1 **{report['test']['pii_f1']:.3f}**, injection F1 **{report['test']['injection_f1']:.3f}**.", "", f"Synthetic validation mean ECE **{report['calibration']['raw_mean']['ece']:.3f} raw / {report['calibration']['calibrated_mean']['ece']:.3f} calibrated**. Mean Brier **{report['calibration']['calibrated_mean']['brier']:.3f}**; NLL **{report['calibration']['calibrated_mean']['nll']:.3f}**. Temperatures for eight heads were fitted on this same validation set.", "", f"Controlled adversarial injection: clean F1 **{report['adversarial']['clean_f1']:.3f}** (24 rows), transformed F1 **{report['adversarial']['transformed_f1']:.3f}** (216 rows). Worst variant: **{worst_variant}**, F1 **{report['adversarial']['worst_variant_f1']:.3f}**. The transformed aggregate improves because some prefixes make attacks easier; it does not prove robustness.", "", f"Separate CC-BY Turkish PII test: v1 F1 **{report['independent_pii']['v1_f1']:.3f}**, v2 F1 **{report['independent_pii']['v2_f1']:.3f}**, v2 FPR **{report['independent_pii']['v2_false_positive_rate']:.3f}**. Source is synthetic and this set was inspected after training; later tuning against it would bias subsequent claims.", "", f"ONNX FP32: **{report['edge']['size_mib']:.1f} MiB**, batch-1 CPU p50 **{report['edge']['onnx_cpu_p50_ms']:.3f} ms**. INT8 final-action disagreement **{report['edge']['int8_final_action_disagreement']:.3f}** on 256 synthetic rows.", "", f"Abstention rate **{report['uncertainty']['abstention_rate']:.3f}**. Confidence median **{evaluation['confidence_distribution']['median']:.3f}**. Selective action error rises as threshold increases; current confidence is not a valid correctness estimate.", "", "See `reports/model_health.json` for coverage-risk, per-task calibration, size, and all exact measured values."]
    if agent:
        one = report["independent_agent_injection"]["single_window"]
        window = report["independent_agent_injection"]["sliding_max"]
        lines += ["", f"Independent Apache-2.0 agentic tool-result benchmark (142 attacks, 40 benign): single-window injection F1 **{one['f1']:.3f}**, recall **{one['recall']:.3f}**, FPR **{one['false_positive_rate']:.3f}**; sliding-window max F1 **{window['f1']:.3f}**, recall **{window['recall']:.3f}**, FPR **{window['false_positive_rate']:.3f}**. This is text detection only, not agent attack success. Neither setting is adequate for autonomous enforcement. See `reports/agent_injection_independent.json`."]
    (ROOT / "reports/model_health.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: report[key] for key in ("test", "edge", "uncertainty")}, indent=2))


if __name__ == "__main__":
    main()
