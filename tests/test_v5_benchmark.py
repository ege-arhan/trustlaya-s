import pytest

from trustlaya.v5_benchmark import bucket, validate_manifest, validate_record
from trustlaya.v5_context import STRATEGIES, aggregate_score, candidate_windows


def row(**changes):
    item = {"sample_id": "a", "source": "source_a", "source_url": "https://example.org/a",
            "original_dataset": "dataset", "language": "en", "provenance_type": "human_game",
            "synthetic": False, "human_generated": True, "augmented": False,
            "source_label": "candidate", "review_status": "UNREVIEWED", "split": "CANDIDATE",
            "text_sha256": "a" * 64, "normalized_sha256": "b" * 64,
            "original_token_length": 100, "model_visible_token_length": 94,
            "truncated": True, "length_bucket": "0-128", "gold_label": None,
            "attack_location": None}
    item.update(changes)
    return item


def test_bucket_boundaries():
    assert [bucket(n) for n in (0, 128, 129, 512, 513, 1536, 1537)] == [
        "0-128", "0-128", "129-256", "385-512", "513-768", "1025-1536", "1537+"]


def test_unreviewed_cannot_be_gold():
    with pytest.raises(ValueError):
        validate_record(row(gold_label="DIRECT_ATTACK"))


def test_source_separation_and_normalized_overlap():
    with pytest.raises(ValueError):
        validate_manifest([row(split="TRAIN"), row(sample_id="b", split="DEV", normalized_sha256="c"*64)])
    with pytest.raises(ValueError):
        validate_manifest([row(split="TRAIN"), row(sample_id="b", source="source_b", split="DEV")])


def test_all_context_strategies_bounded():
    for strategy in STRATEGIES:
        windows = candidate_windows(list(range(400)), strategy, cues=[200])
        assert windows and all(0 < len(w) <= 94 for w in windows)
        assert 0 <= aggregate_score([0.1] * len(windows), strategy) <= 1


def test_router_selects_cue_region():
    windows = candidate_windows(list(range(400)), "TWO_STAGE_CONTEXT_ROUTER", cues=[200])
    assert len(windows) == 3
    assert any(200 in window for window in windows)


def test_max_vs_mean():
    assert aggregate_score([.1, .9], "MAX_WINDOW_SCORE") == .9
    assert aggregate_score([.1, .9], "MEAN_WINDOW_SCORE") == .5
