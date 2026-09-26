"""Jev SILVER plumbing; no external calls or claims of human correctness."""
import json
from pathlib import Path

import pytest

from scripts.run_v5_jev_silver import original_permitted_rows, project, run


def answers(intent="BENIGN_DUAL_USE", quoted=0.9, directed=0.1):
    return {
        "intent": {"choice": intent, "probabilities": {"ATTACK": 0.1,
                   "BENIGN_DUAL_USE": 0.8, "NORMAL": 0.05, "UNRESOLVED": 0.05}},
        "attack_vector": {"choice": "DIRECT_OVERRIDE"},
        "benign_type": {"choice": "QUOTED_REFERENCE"},
        "directed_attack": {"probability": directed},
        "quoted_reference": {"probability": quoted},
        "contains_pii": {"probability": 0.1},
        "contains_secret": {"probability": 0.1},
        "obfuscated": {"probability": 0.1},
    }


PRIVATE_PACKET = Path(__file__).resolve().parents[1] / "benchmarks/v5/private/review_packet.jsonl"
local_packet = pytest.mark.skipif(not PRIVATE_PACKET.exists(), reason="private review packet is local-only")


@local_packet
def test_original_packet_excludes_unlicensed_tensor_trust():
    rows, digest = original_permitted_rows()
    assert len(rows) == 260
    assert len(digest) == 64
    assert all(row["license"] != "unspecified" for row in rows)
    assert all(row["source"] != "tensor_trust_game" for row in rows)


def test_quoted_attack_is_benign_silver():
    result = project(answers())
    assert result["intent"] == "BENIGN_DUAL_USE"
    assert result["benign_type"] == "QUOTED_REFERENCE"
    assert result["annotation_source"] == "JEV_SILVER"


def test_attack_conflict_is_flagged():
    result = project(answers("ATTACK"))
    assert "ATTACK_QUOTE_CONFLICT" in result["quality_flags"]
    assert "ATTACK_INTENT_CONFLICT" in result["quality_flags"]


def test_unknown_intent_is_rejected():
    with pytest.raises(ValueError, match="unknown intent"):
        project(answers("WRONG"))


def test_replacement_uses_train_only(tmp_path, monkeypatch):
    from scripts import run_v5_jev_silver as module
    import datasets

    def fake_dataset(name, split, revision):
        assert name == "deepset/prompt-injections"
        assert split == "train"
        assert revision == module.DEEPSET_REVISION
        return [{"label": i % 2, "text": f"unique example {i}"} for i in range(546)]

    monkeypatch.setattr(datasets, "load_dataset", fake_dataset)
    rows, digest = module.replacement_rows([], tmp_path)
    assert len(rows) == 340
    assert {r["source_proxy_label"] for r in rows} == {0, 1}
    assert len({r["text_sha256"] for r in rows}) == 340
    assert len(digest) == 64


def test_resume_and_hash_guard(tmp_path, monkeypatch):
    from scripts import run_v5_jev_silver as module
    calls = []

    def fake_call(text, key):
        calls.append(text)
        return {"model": "typesafe-ai/jev", "answers": answers(), "usage": {}}

    monkeypatch.setattr(module, "call_jev", fake_call)
    row = {"sample_id": "one", "text": "test", "source": "test", "source_url": "https://example.test",
           "license": "MIT", "text_sha256": module.sha256(b"test")}
    output = tmp_path / "silver.jsonl"
    assert len(run([row], output, "dummy")) == 1
    assert len(run([row], output, "dummy")) == 1
    assert calls == ["test"]
    changed = dict(row, text_sha256=module.sha256(b"other"))
    with pytest.raises(ValueError, match="hash mismatch"):
        run([changed], output, "dummy")
    assert json.loads(output.read_text())["silver"]["annotation_source"] == "JEV_SILVER"


@local_packet
def test_manifest_refuses_partial_results(tmp_path):
    from scripts.summarize_v5_jev_silver import summarize

    output = tmp_path / "partial.jsonl"
    output.write_text(json.dumps({"sample_id": "one"}) + "\n")
    with pytest.raises(ValueError, match="600 unique"):
        summarize(output)
