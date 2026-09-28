from scripts.evaluate_v5_jev_silver import compare, threshold_sweep


def test_silver_agreement_excludes_unresolved_and_keeps_frozen_threshold():
    rows = [
        {"jev_binary": 1, "v2_attack": 1, "v2_raw_score": 0.9},
        {"jev_binary": 0, "v2_attack": 1, "v2_raw_score": 0.6},
        {"jev_binary": 0, "v2_attack": 0, "v2_raw_score": 0.2},
        {"jev_binary": None, "v2_attack": 1, "v2_raw_score": 0.99},
    ]
    result = compare(rows, "jev_binary")
    assert (result["n"], result["tp"], result["fp"], result["tn"], result["fn"]) == (3, 1, 1, 1, 0)
    sweep = threshold_sweep(rows, "jev_binary")
    assert (sweep[50]["tp"], sweep[50]["fp"]) == (1, 1)
    assert (sweep[70]["tp"], sweep[70]["fp"]) == (1, 0)
