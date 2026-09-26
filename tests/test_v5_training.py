"""V5 training pipeline invariants: data selection, sampling determinism, loss, checkpoints."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import BertConfig

from trustlaya.v5_model import MAX_CONTENT, V5Classifier, batch, focal_loss, source_balanced_weights

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_v5_ablation import EXPERIMENTS, epoch_order  # noqa: E402


def lineage_rows():
    data = json.loads((ROOT / "data/dataset_lineage.json").read_text())
    return [dict(zip(data["fields"], row)) for row in data["rows"]]


@pytest.mark.parametrize("experiment_id", sorted(EXPERIMENTS))
def test_training_rows_are_train_split_only(experiment_id):
    manifest = ROOT / "reports/experiments" / experiment_id / "dataset_manifest.json"
    if not manifest.exists():
        pytest.skip("experiment not trained yet")
    sources = set(EXPERIMENTS[experiment_id]["sources"])
    ids = sorted(r["sample_id"] for r in lineage_rows() if r["status"] == "included" and r["split"] == "TRAIN"
                 and r["source"] in sources and r["label"] in ("ATTACK", "BENIGN"))
    stored = json.loads(manifest.read_text())["train_sample_ids_sha256"]
    assert hashlib.sha256("\n".join(ids).encode()).hexdigest() == stored


def test_sampling_order_is_seeded():
    weights = source_balanced_weights([("tt", "ATTACK")] * 5 + [("jll", "ATTACK")] + [("jll", "BENIGN")] * 3)
    assert epoch_order(9, weights, 0) == epoch_order(9, weights, 0)
    assert epoch_order(9, None, 1) == epoch_order(9, None, 1)
    assert epoch_order(9, None, 1) != epoch_order(9, None, 2)


def test_balanced_weights_split_mass_by_label_then_source():
    groups = [("tt", "ATTACK")] * 8 + [("jll", "ATTACK")] * 2 + [("jll", "BENIGN")] * 4 + [("docs", "BENIGN")]
    weights = source_balanced_weights(groups)
    mass = {}
    for g, w in zip(groups, weights):
        mass[g] = mass.get(g, 0) + w
    assert mass[("tt", "ATTACK")] == pytest.approx(0.25) and mass[("jll", "ATTACK")] == pytest.approx(0.25)
    assert mass[("jll", "BENIGN")] == pytest.approx(0.25) and mass[("docs", "BENIGN")] == pytest.approx(0.25)


def test_focal_loss_reduces_to_weighted_bce():
    logits, targets = torch.tensor([0.3, -1.2, 2.0]), torch.tensor([1.0, 0.0, 1.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert focal_loss(logits, targets, gamma=0.0, alpha=0.5).item() == pytest.approx(0.5 * bce.item())


def test_context_limit_is_510_content_tokens():
    ids, mask = batch([list(range(5, 800))], MAX_CONTENT, 2, 3)
    assert ids.shape[1] == 512 and ids[0, 0] == 2 and ids[0, -1] == 3
    with pytest.raises(ValueError):
        batch([[5]], 512, 2, 3)


def test_checkpoint_round_trip(tmp_path):
    config = BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=32)
    config.save_pretrained(tmp_path)
    torch.manual_seed(0)
    model = V5Classifier(str(tmp_path), pretrained=False).eval()
    model.save(tmp_path / "model.safetensors")
    loaded = V5Classifier(str(tmp_path), pretrained=False).eval()
    loaded.load(tmp_path / "model.safetensors")
    ids, mask = batch([[5, 6, 7], [8]], 10, 2, 3)
    assert torch.equal(model(ids, mask), loaded(ids, mask))


def test_tokenizer_matches_recorded_checksum():
    tokenizer = ROOT / "models/trustlaya-s-v2/tokenizer.json"
    configs = sorted((ROOT / "reports/experiments").glob("*/config.json"))
    if not tokenizer.exists() or not configs:
        pytest.skip("local tokenizer or experiment configs absent")
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    assert all(json.loads(c.read_text())["tokenizer_sha256"] == digest for c in configs)
