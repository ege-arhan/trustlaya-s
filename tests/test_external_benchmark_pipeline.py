import numpy as np

from trustlaya.v3_external import (
    BIO, binary_metrics, calibration_metrics, canonicalize_spans,
    decode_spans, fingerprint, span_metrics, token_labels,
)


def test_bio_label_mapping_matches_tab_direct_categories():
    offsets = [(0, 0), (0, 2), (2, 5), (6, 10), (0, 0)]
    assert token_labels(offsets, [(0, 5, "PERSON"), (6, 10, "CODE")]) == [0, 1, 2, 3, 0]
    assert len(BIO) == 5


def test_duplicate_gold_span_does_not_create_overlapping_bio_error():
    assert token_labels([(0, 4)], [(0, 4, "CODE"), (0, 4, "CODE")]) == [3]


def test_unknown_tab_entity_is_not_silently_mapped():
    import pytest
    with pytest.raises(ValueError):
        token_labels([(0, 3)], [(0, 3, "EMAIL")])


def test_subword_b_tags_can_form_one_span():
    assert decode_spans([3, 3, 4], [(0, 2), (2, 5), (5, 8)]) == [(0, 8, "CODE")]


def test_application_number_offsets_trim_punctuation():
    text = "(no. 40593/04)"
    assert canonicalize_spans(text, [(5, 14, "CODE")]) == [(5, 13, "CODE")]


def test_exact_span_metrics_do_not_equal_presence_metrics():
    result = span_metrics([[(0, 4, "PERSON")]], [[(0, 5, "PERSON")]])
    assert result["ALL"]["f1"] == 0
    assert binary_metrics(np.array([1]), np.array([.9]), .5)["f1"] == 1


def test_normalized_duplicate_hash():
    assert fingerprint("  Hello  WORLD ") == fingerprint("hello world")


def test_threshold_metrics_count_false_alarms_and_misses():
    m = binary_metrics(np.array([0, 0, 1, 1]), np.array([.1, .8, .2, .9]), .5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["fpr"] == m["fnr"] == .5


def test_calibration_metrics_distinguish_well_and_overconfident_predictions():
    y = np.array([0, 1]); good = calibration_metrics(y, np.array([.1, .9]))
    bad = calibration_metrics(y, np.array([.9, .1]))
    assert good["brier"] < bad["brier"]
    assert good["nll"] < bad["nll"]
