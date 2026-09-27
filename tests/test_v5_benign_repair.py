"""Benign-repair data and sampling invariants (no raw text needed)."""

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_v5_benign_repair import NEW_SOURCES, grouped_batches, weights_for  # noqa: E402


def rows():
    data = json.loads((ROOT / "data/benign_repair_lineage.json").read_text())
    return [dict(zip(data["fields"], r)) for r in data["rows"]]


def test_manifest_checksums_match():
    for line in (ROOT / "data/benign_repair_manifest.sha256").read_text().splitlines():
        digest, path = line.split()
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path


def test_ood_sources_never_enter_train_or_dev():
    for r in rows():
        if r["source"] == "massive":
            assert r["split"] == "OOD_BENIGN_TEST"
        assert r["source"] in NEW_SOURCES + ("massive",)  # deepset is not part of this dataset


def test_new_rows_never_share_a_cluster_across_splits():
    splits = defaultdict(set)
    for r in rows():
        if r["status"] == "included":
            splits[r["cluster"]].add(r["split"])
    assert all(len(s) == 1 for s in splits.values())


def test_weight_multiplier_keeps_class_mass_and_scales_new_sources():
    train = ([{"source": "tensor_trust", "label": "ATTACK"}] * 4 + [{"source": "jailbreakllms", "label": "BENIGN"}] * 2
             + [{"source": "dolly", "label": "BENIGN"}] * 2)
    base = weights_for(train, {"sampling": "BALANCED_SOURCE", "new_weight": 1})
    four = weights_for(train, {"sampling": "BALANCED_SOURCE", "new_weight": 4})
    mass = lambda w, pred: sum(x for r, x in zip(train, w) if pred(r))
    for w in (base, four):
        assert mass(w, lambda r: r["label"] == "ATTACK") == pytest.approx(mass(w, lambda r: r["label"] == "BENIGN"))
    new_share = lambda w: mass(w, lambda r: r["source"] == "dolly") / mass(w, lambda r: r["source"] == "jailbreakllms")
    assert new_share(four) == pytest.approx(4 * new_share(base))
    assert weights_for(train, {"sampling": "NATURAL", "new_weight": 1}) is None


def test_grouped_batches_are_a_seeded_partition():
    order = list(range(1000))
    lengths = [i % 97 for i in order]
    first = grouped_batches(order, lengths, 16, 0)
    assert sorted(i for b in first for i in b) == order
    assert first == grouped_batches(order, lengths, 16, 0)
    assert first != grouped_batches(order, lengths, 16, 1)


def test_iter3_manifest_and_roles():
    for line in (ROOT / "data/iter3_manifest.sha256").read_text().splitlines():
        digest, path = line.split()
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    data = json.loads((ROOT / "data/iter3_lineage.json").read_text())
    for r in (dict(zip(data["fields"], row)) for row in data["rows"]):
        if r["source"] == "massive_dev":
            assert r["split"] == "OOD_BENIGN_FRESH"
        else:
            assert r["source"] == "hackaprompt" and r["native_category"].startswith("successful_attack")
            assert r["split"] in ("TRAIN", "DEV", "TEST")
