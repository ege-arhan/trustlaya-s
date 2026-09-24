"""Checks for external-evaluation bookkeeping, without downloading datasets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from run_external_real import contamination, metrics, norm


def test_metrics_binary_confusion_and_calibration():
    rows=[{"gold":1,"raw_score":.9},{"gold":1,"raw_score":.2},
          {"gold":0,"raw_score":.7},{"gold":0,"raw_score":.1}]
    m=metrics(rows)
    assert (m["tp"],m["fp"],m["tn"],m["fn"])==(1,1,1,1)
    assert m["precision"]==m["recall"]==m["f1"]==.5
    assert m["fpr"]==m["fnr"]==.5
    assert 0 <= m["ece"] <= 1


def test_contamination_excludes_exact_near_and_benchmark_repeats():
    train=["A customer record says the account number is 12345678 and should be private"]
    samples=[{"text":train[0].upper(),"gold":1},
             {"text":"A customer record says the account number is 12345678 and should be private!","gold":1},
             {"text":"This unrelated passage describes weather over the mountains","gold":0},
             {"text":"This unrelated passage describes weather over the mountains","gold":0}]
    clean,counts=contamination(samples,train)
    assert len(clean)==1
    assert counts["training_exact"]==1
    assert counts["training_near_0.85"]==1
    assert counts["within_benchmark_duplicate"]==1
    assert len(clean[0]["text_hash"])==64


def test_contamination_drops_conflicting_gold_labels():
    samples=[{"text":"The same text", "gold":0},{"text":"The same text", "gold":1},
             {"text":"Other distinct text", "gold":0}]
    clean,counts=contamination(samples,["Unrelated train text"])
    assert len(clean)==1 and clean[0]["text"]=="Other distinct text"
    assert counts["label_conflict_removed"]==2
