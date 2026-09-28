"""Exact, normalized and character n-gram near-duplicate screen for v3."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer

from run_external_real import LOCAL, tab_samples, jailbreak_samples
from train_v3_external import ROOT, benign_prompts, gandalf
from trustlaya.v3_external import normalized


def audit(reference, target, target_ids, cutoff=.85):
    norm_ref = [normalized(t) for t in reference]
    norm_target = [normalized(t) for t in target]
    exact = set(norm_ref)
    v = HashingVectorizer(analyzer="char", ngram_range=(4, 5), n_features=2**18,
                          alternate_sign=False, norm="l2")
    matrix = v.transform(norm_ref)
    query = v.transform(norm_target)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1).fit(matrix)
    distances, _ = nn.kneighbors(query)
    sim = 1 - distances[:, 0]
    overlaps = []
    for ident, text, similarity in zip(target_ids, norm_target, sim, strict=True):
        if text in exact or similarity >= cutoff:
            overlaps.append({"sample_id": ident, "exact": text in exact, "similarity": round(float(similarity), 5)})
    return {"reference_n": len(reference), "target_n": len(target), "exact_n": sum(x["exact"] for x in overlaps),
            "near_or_exact_n": len(overlaps), "cutoff": cutoff, "overlaps": overlaps}


def main():
    tok = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v3")
    train, _ = tab_samples(tok, "train"); dev, _ = tab_samples(tok, "dev"); test, _ = tab_samples(tok, "test")
    frozen = {r["sample_id"] for r in json.loads((LOCAL / "v2_frozen_tab_predictions.json").read_text())}
    test = [r for r in test if r["sample_id"] in frozen]
    benign, _ = benign_prompts()
    attack_train = gandalf("train") + benign["train"]
    attack_dev = gandalf("validation") + benign["validation"]
    gandalf_test = gandalf("test")
    jailbreak = jailbreak_samples()[0]
    frozen_jail = {r["sample_id"] for r in json.loads((LOCAL / "v2_frozen_jailbreak_predictions.json").read_text())}
    jailbreak = [r for r in jailbreak if r["sample_id"] in frozen_jail]
    pairs = {
        "TAB_train_dev": (train, dev), "TAB_train_test": (train, test), "TAB_dev_test": (dev, test),
    }
    result = {}
    for name, (left, right) in pairs.items():
        result[name] = audit([r["text"] for r in left], [r["text"] for r in right], [r["sample_id"] for r in right])
        print(name, {k: v for k, v in result[name].items() if k != "overlaps"}, flush=True)
    result["prompt_train_dev"] = audit(attack_train, attack_dev, [f"dev:{i}" for i in range(len(attack_dev))])
    print("prompt_train_dev", {k: v for k, v in result["prompt_train_dev"].items() if k != "overlaps"}, flush=True)
    result["prompt_train_jailbreak_test"] = audit(attack_train, [r["text"] for r in jailbreak], [r["sample_id"] for r in jailbreak])
    print("prompt_train_jailbreak_test", {k: v for k, v in result["prompt_train_jailbreak_test"].items() if k != "overlaps"}, flush=True)
    result["prompt_dev_jailbreak_test"] = audit(attack_dev, [r["text"] for r in jailbreak], [r["sample_id"] for r in jailbreak])
    print("prompt_dev_jailbreak_test", {k: v for k, v in result["prompt_dev_jailbreak_test"].items() if k != "overlaps"}, flush=True)
    result["prompt_train_gandalf_test"] = audit(attack_train, gandalf_test, [f"gandalf:{i}" for i in range(len(gandalf_test))])
    result["prompt_dev_gandalf_test"] = audit(attack_dev, gandalf_test, [f"gandalf:{i}" for i in range(len(gandalf_test))])
    print("Gandalf test overlap", result["prompt_train_gandalf_test"]["near_or_exact_n"],
          result["prompt_dev_gandalf_test"]["near_or_exact_n"], flush=True)
    (ROOT / "reports/v3_leakage_results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__": main()
