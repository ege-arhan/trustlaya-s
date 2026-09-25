"""Freeze the exact v2/v3 comparison inputs before any v4 experiment.

Copies row predictions and metrics to a local read-only snapshot. The tracked
manifest and report contain checksums so silent replacement is detectable.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/external/predictions"
FROZEN = SOURCE / "v4_frozen_inputs"
MANIFEST = ROOT / "reports/v4_frozen_manifest.json"
REPORT = ROOT / "reports/v2_v3_frozen_comparison.md"
FILES = {
    "v2_tab_predictions.json": SOURCE / "v2_frozen_tab_predictions.json",
    "v2_jailbreak_predictions.json": SOURCE / "v2_frozen_jailbreak_predictions.json",
    "v3_tab_predictions.json": SOURCE / "v3_tab_predictions.json",
    "v3_jailbreak_predictions.json": SOURCE / "v3_jailbreak_predictions.json",
    "v2_metrics.json": SOURCE / "v2_frozen_external_results.json",
    "v3_metrics.json": ROOT / "reports/v3_external_results.json",
    "v3_exclusions.json": ROOT / "reports/v3_leakage_results.json",
    "v2_calibration.json": ROOT / "models/trustlaya-s-v2/calibration.json",
    "v2_thresholds.json": ROOT / "models/trustlaya-s-v2/decision_thresholds.json",
    "v3_calibration.json": ROOT / "models/trustlaya-s-v3/calibration.json",
    "v3_thresholds.json": ROOT / "models/trustlaya-s-v3/decision_thresholds.json",
    "v2_policy.yaml": ROOT / "models/trustlaya-s-v2/policy.yaml",
    "v3_model_manifest.json": ROOT / "models/trustlaya-s-v3/manifest.json",
    "encoder_config.json": ROOT / "models/base/config.json",
}
WEIGHTS = {
    "v2_model": ROOT / "models/trustlaya-s-v2/model.safetensors",
    "v3_pii_head": ROOT / "models/trustlaya-s-v3/pii_token_head.safetensors",
    "v3_attack_head": ROOT / "models/trustlaya-s-v3/attack_head.safetensors",
    "v2_tokenizer": ROOT / "models/trustlaya-s-v2/tokenizer.json",
    "v3_tokenizer": ROOT / "models/trustlaya-s-v3/tokenizer.json",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    FROZEN.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, source in FILES.items():
        assert source.is_file(), f"Missing baseline artifact: {source}"
        target = FROZEN / name
        digest = sha(source)
        if target.exists():
            assert sha(target) == digest, f"Frozen artifact changed: {target}"
        else:
            shutil.copyfile(source, target)
            target.chmod(0o444)
        hashes[name] = digest
    weights = {name: sha(path) for name, path in WEIGHTS.items()}
    assert weights["v2_model"] == "99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c"
    source_revisions = {
        name: subprocess.check_output(["git", "-C", str(ROOT / "benchmarks/external/raw" / name),
                                       "rev-parse", "HEAD"], text=True).strip()
        for name in ("tab", "jailbreakllms")
    }
    assert source_revisions == {
        "tab": "558e09e26d6b36f5f78440074e6a233946d98bd9",
        "jailbreakllms": "2dbd7bbc25f1b156552678f451bddbc787cd679f",
    }
    v2_tab = json.loads((FROZEN / "v2_tab_predictions.json").read_text())
    v3_tab = json.loads((FROZEN / "v3_tab_predictions.json").read_text())
    v2_jail = json.loads((FROZEN / "v2_jailbreak_predictions.json").read_text())
    v3_jail = json.loads((FROZEN / "v3_jailbreak_predictions.json").read_text())
    assert (len(v2_tab), len(v3_tab), len(v2_jail), len(v3_jail)) == (2079, 1794, 5888, 5761)
    state = {"frozen_from_commit": "508483f", "file_sha256": hashes,
             "weight_and_tokenizer_sha256": weights, "dataset_revisions": source_revisions,
             "row_counts": {"v2_TAB": 2079, "v3_TAB_clean": 1794,
                            "v2_JailbreakLLMs": 5888, "v3_JailbreakLLMs_clean": 5761}}
    payload = json.dumps(state, indent=2) + "\n"
    if MANIFEST.exists():
        assert MANIFEST.read_text() == payload, "Frozen tracked manifest changed"
    else:
        MANIFEST.write_text(payload)
    table = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in sorted(hashes.items()))
    report = f"""# Frozen v2/v3 baseline for v4

This snapshot was made from commit `508483f` before v4 model or data experiments. The exact row predictions, metrics, calibration and threshold files are copied into the gitignored local `benchmarks/external/predictions/v4_frozen_inputs/`. Those copies are read-only. The tracked [manifest](v4_frozen_manifest.json) and table below provide SHA-256 verification. Model and tokenizer files are referenced by checksum; neither is overwritten.

## Fixed paired external comparisons

| Dataset / granular task | Clean N | Version | Precision | Recall | F1 | FPR | Threshold / calibration |
|---|---:|---|---:|---:|---:|---:|---|
| TAB DIRECT PERSON/CODE window presence | 1,794 | v2 frozen | 0.115 | 0.165 | 0.135 | 0.077 | 0.50 raw |
| TAB DIRECT PERSON/CODE window presence | 1,794 | v3 NO-GO | 0.306 | 0.107 | 0.158 | 0.015 | 0.47 Platt-calibrated; token threshold 0.50 raw |
| JailbreakLLMs community direct-jailbreak proxy | 5,761 | v2 frozen | 0.115 | 0.946 | 0.206 | 0.898 | 0.50 raw |
| JailbreakLLMs community direct-jailbreak proxy | 5,761 | v3 NO-GO | 0.076 | 0.035 | 0.048 | 0.052 | 0.30 Platt-calibrated |

The full original v2 cohorts remain 2,079 TAB and 5,888 JailbreakLLMs rows. The smaller paired cohorts exclude the exact/near overlaps listed in `v3_exclusions.json`. V2/v3 exact predictions are not regenerated from a changed pipeline for this comparison. TAB is window presence here; v3 exact-span F1 was 0.079 and must not be mixed with this table.

## Fixed source versions and models

- TAB Git revision `{source_revisions['tab']}` (official ECHR test).
- JailbreakLLMs Git revision `{source_revisions['jailbreakllms']}` (Reddit/Discord/website subset).
- v2 model SHA-256 `{weights['v2_model']}`.
- v3 PII head SHA-256 `{weights['v3_pii_head']}`; attack head SHA-256 `{weights['v3_attack_head']}`; the v2 encoder is shared and frozen.
- v2 tokenizer SHA-256 `{weights['v2_tokenizer']}`; v3 tokenizer SHA-256 `{weights['v3_tokenizer']}`.
- Detailed configuration, calibration and threshold JSONs are in the snapshot below.

| Snapshot file | SHA-256 |
|---|---|
{table}

`python scripts/freeze_v4_baselines.py` verifies this snapshot on future runs and refuses mismatched copies. External-test reuse after v3 iteration means the old JailbreakLLMs set is a frozen comparison, **not a new blind v4 final test**. V4 requires a separately sourced unseen final set.
"""
    if REPORT.exists():
        assert REPORT.read_text() == report, "Frozen report changed"
    else:
        REPORT.write_text(report)
    print("Frozen v2/v3 inputs verified:", FROZEN)


if __name__ == "__main__": main()
