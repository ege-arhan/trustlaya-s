"""Freeze v2/v3/v4 comparison inputs without regenerating predictions."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private/frozen"
REPORT = ROOT / "reports/v5_frozen_baselines.md"
MANIFEST = ROOT / "benchmarks/v5/frozen_manifest.json"

FILES = {
    "v2_jll_predictions.json": ROOT / "benchmarks/external/predictions/v4_frozen_inputs/v2_jailbreak_predictions.json",
    "v3_jll_predictions.json": ROOT / "benchmarks/external/predictions/v4_frozen_inputs/v3_jailbreak_predictions.json",
    "v4_jll_predictions.json": ROOT / "benchmarks/external/predictions/v4_jailbreak_final_predictions.json",
    "v2_tab_predictions.json": ROOT / "benchmarks/external/predictions/v4_frozen_inputs/v2_tab_predictions.json",
    "v3_tab_predictions.json": ROOT / "benchmarks/external/predictions/v4_frozen_inputs/v3_tab_predictions.json",
    "v2_v3_manifest.json": ROOT / "reports/v4_frozen_manifest.json",
    "v4_jll_metrics.json": ROOT / "reports/v4_final_external_metrics.json",
    "v4_deepset_metrics.json": ROOT / "reports/v4_unseen_external_metrics.json",
    "v4_inference_config.json": ROOT / "models/trustlaya-s-v4-research/manifest.json",
    "v4_operating_point.json": ROOT / "models/trustlaya-s-v4-research/operating_point.json",
    "v2_calibration.json": ROOT / "models/trustlaya-s-v2/calibration.json",
    "v3_calibration.json": ROOT / "models/trustlaya-s-v3/calibration.json",
    "v2_thresholds.json": ROOT / "models/trustlaya-s-v2/decision_thresholds.json",
    "v3_thresholds.json": ROOT / "models/trustlaya-s-v3/decision_thresholds.json",
    "encoder_config.json": ROOT / "models/base/config.json",
}
WEIGHTS = {
    "v2_model": ROOT / "models/trustlaya-s-v2/model.safetensors",
    "v3_attack_head": ROOT / "models/trustlaya-s-v3/attack_head.safetensors",
    "v3_pii_head": ROOT / "models/trustlaya-s-v3/pii_token_head.safetensors",
    "v4_attack_head": ROOT / "models/trustlaya-s-v4-research/attack_intent_head.safetensors",
    "tokenizer": ROOT / "models/trustlaya-s-v2/tokenizer.json",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for name, source in FILES.items():
        digest = sha(source)
        target = PRIVATE / name
        if target.exists():
            if sha(target) != digest:
                raise ValueError(f"frozen file changed: {name}")
        else:
            shutil.copyfile(source, target)
            target.chmod(0o444)
        file_hashes[name] = digest
    weight_hashes = {name: sha(path) for name, path in WEIGHTS.items()}
    previous = json.loads((ROOT / "reports/v4_frozen_manifest.json").read_text())
    assert weight_hashes["v2_model"] == previous["weight_and_tokenizer_sha256"]["v2_model"]
    assert weight_hashes["v3_attack_head"] == previous["weight_and_tokenizer_sha256"]["v3_attack_head"]
    assert weight_hashes["tokenizer"] == previous["weight_and_tokenizer_sha256"]["v2_tokenizer"]
    config = json.loads(FILES["v4_inference_config.json"].read_text())
    assert weight_hashes["v4_attack_head"] == config["head_sha256"]
    snapshot = {
        "frozen_from_commit": "d8272bc",
        "model_sha256": weight_hashes,
        "file_sha256": file_hashes,
        "dataset_revisions": previous["dataset_revisions"],
        "benchmark": "JailbreakLLMs clean paired cohort, N=5761; TAB prior cohorts; deepset separate task",
        "operating_points": {"v2": {"threshold": 0.5, "calibration": "raw", "max_total_tokens": 96},
                             "v3": {"threshold": 0.3, "calibration": "Platt", "max_total_tokens": 96},
                             "v4": {"threshold": 0.79, "calibration": "Platt after WINDOW_MAX", "window_total_tokens": 96}},
    }
    serialized = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and MANIFEST.read_text() != serialized:
        raise ValueError("tracked frozen manifest changed")
    MANIFEST.write_text(serialized)
    report = """# V5 starting point: frozen v2/v3/v4

Exact row predictions and configuration are copied to local read-only `benchmarks/v5/private/frozen/`. The tracked `benchmarks/v5/frozen_manifest.json` stores SHA-256 values. The script rejects changed copies and checks the v2/v3 original freeze. No model or previous report is rewritten.

| Version | JLL clean N | F1 | Recall | FPR | Operating point |
|---|---:|---:|---:|---:|---|
| v2 default | 5,761 | 0.205857 | 0.946457 | 0.897971 | raw 0.50; first 94 content tokens |
| v3 NO-GO | 5,761 | 0.047619 | 0.034646 | 0.052087 | Platt 0.30; first 94 content tokens |
| v4 NO-GO | 5,761 | 0.249945 | 0.900787 | 0.657433 | window max, Platt 0.79 |

V4 has 0.860281 FPR on benign inputs above 510 content tokens. On the separate deepset prompt-injection task it detected 4/60 positives. These are distinct task definitions. JLL was previously inspected, so it is a frozen historical comparison, not a new blind V5 test. The v4 model is local and does not replace v2.
"""
    if REPORT.exists() and REPORT.read_text() != report:
        raise ValueError("frozen report changed")
    REPORT.write_text(report)
    print(f"Verified {len(FILES)} frozen files and {len(WEIGHTS)} model/tokenizer hashes")


if __name__ == "__main__":
    main()
