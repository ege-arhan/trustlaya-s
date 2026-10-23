import pytest
import csv
import json

from scripts import adjudicate_v5_reviews as reviews

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
    with pytest.raises(ValueError):
        validate_record(row(gold_label="AMBIGUOUS", review_status="AGREED"))


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


def test_krippendorff_nominal_alpha():
    assert reviews.krippendorff_alpha_nominal([]) is None
    assert reviews.krippendorff_alpha_nominal([("ATTACK", "ATTACK")]) is None
    assert reviews.krippendorff_alpha_nominal([("ATTACK", "ATTACK"), ("NORMAL", "NORMAL")]) == 1
    assert reviews.krippendorff_alpha_nominal([("ATTACK", "NORMAL")]) == 0


def _review_csv(path, annotator, intent):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=reviews.FIELDS)
        writer.writeheader()
        writer.writerow({"sample_id": "r-blind", "annotator_id": annotator, "intent": intent,
                         "attack_vector": "DIRECT_OVERRIDE" if intent == "ATTACK" else "",
                         "benign_type": "", "language": "en", "attack_location": "",
                         "contains_pii": "no", "contains_secret": "no", "obfuscated": "no"})


def test_unresolved_review_never_becomes_gold(tmp_path, monkeypatch):
    manifest = tmp_path / "benchmarks/v5/dataset_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(row()) + "\n")
    private = tmp_path / "benchmarks/v5/private"
    private.mkdir()
    (private / "review_id_map.json").write_text('{"r-blind":"a"}')
    monkeypatch.setattr(reviews, "ROOT", tmp_path)
    monkeypatch.setattr(reviews, "PRIVATE", private)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _review_csv(a, "human_a", "UNRESOLVED")
    _review_csv(b, "human_b", "UNRESOLVED")
    result = reviews.adjudicate(a, b, None)
    assert result["gold_n"] == 0
    assert result["ready_for_freeze"] is False
    assert "gold_intent" not in json.loads((private / "reviewed_manifest.jsonl").read_text())


def test_adjudicator_must_be_third_person(tmp_path, monkeypatch):
    manifest = tmp_path / "benchmarks/v5/dataset_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(row()) + "\n")
    private = tmp_path / "benchmarks/v5/private"
    private.mkdir()
    (private / "review_id_map.json").write_text('{"r-blind":"a"}')
    monkeypatch.setattr(reviews, "ROOT", tmp_path)
    monkeypatch.setattr(reviews, "PRIVATE", private)
    a, b, senior = (tmp_path / name for name in ("a.csv", "b.csv", "senior.csv"))
    _review_csv(a, "human_a", "ATTACK")
    _review_csv(b, "human_b", "NORMAL")
    _review_csv(senior, "human_a", "ATTACK")
    with pytest.raises(ValueError, match="independent"):
        reviews.adjudicate(a, b, senior)


def test_senior_can_resolve_disagreement(tmp_path, monkeypatch):
    manifest = tmp_path / "benchmarks/v5/dataset_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(row()) + "\n")
    private = tmp_path / "benchmarks/v5/private"
    private.mkdir()
    (private / "review_id_map.json").write_text('{"r-blind":"a"}')
    monkeypatch.setattr(reviews, "ROOT", tmp_path)
    monkeypatch.setattr(reviews, "PRIVATE", private)
    a, b, senior = (tmp_path / name for name in ("a.csv", "b.csv", "senior.csv"))
    _review_csv(a, "human_a", "ATTACK")
    _review_csv(b, "human_b", "NORMAL")
    _review_csv(senior, "human_c", "ATTACK")
    result = reviews.adjudicate(a, b, senior)
    assert result["gold_n"] == 1
    reviewed = json.loads((private / "reviewed_manifest.jsonl").read_text())
    assert reviewed["gold_intent"] == "ATTACK"
    assert reviewed["review_status"] == "ADJUDICATED"
