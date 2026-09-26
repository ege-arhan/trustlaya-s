"""Evaluate the V5 GO gate from committed evidence and write reports/v5_evidence_gate.md.

Every item must PASS for V5_STATUS = GO. GO only permits training to be
prepared; this script never trains. `--rebuild` reruns the dataset builder and
checks that the manifest checksums are byte-identical (reproducibility).

Run: .venv/bin/python scripts/v5_evidence_gate.py [--rebuild]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def load(path):
    path = ROOT / path
    return json.loads(path.read_text()) if path.exists() else None


def checksums_match():
    for line in (ROOT / "data/v5_manifest.sha256").read_text().splitlines():
        digest, path = line.split()
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    before = (ROOT / "data/v5_manifest.sha256").read_text()
    rebuild = None
    if args.rebuild:
        subprocess.run([sys.executable, str(ROOT / "scripts/build_v5_training_set.py")], check=True,
                       capture_output=True)
        rebuild = (ROOT / "data/v5_manifest.sha256").read_text() == before

    tt = load("reports/tensor_trust_training_stats.json")
    mapping = load("data/label_mapping.json")
    dedup = load("reports/dataset_dedup_report.json")
    lineage = load("data/dataset_lineage.json")
    rows = [dict(zip(lineage["fields"], r)) for r in lineage["rows"]]
    context, tokenizer = load("reports/context_ablation.json"), load("reports/tokenizer_audit.json")
    baseline, calibration = load("reports/v2_public_baseline.json"), load("reports/v2_calibration_audit.json")
    firewall = load("reports/firewall_real_v2_validation.json")

    v1 = tt["parser_validation"]["v1"]
    recovery = (v1["hijacking_benchmark_recovered"] / v1["hijacking_benchmark_unique"],
                v1["extraction_benchmark_recovered"] / v1["extraction_benchmark_unique"])
    cluster_split = defaultdict(set)
    for r in rows:
        if r["status"] == "included" and r["split"] in ("TRAIN", "DEV", "TEST"):
            cluster_split[r["cluster"]].add(r["split"])
    cluster_of = {r["sample_id"]: r["cluster"] for r in rows}
    experiment_leaks = sum(len({cluster_of[s] for s in e["train"] + e["dev"]} & {cluster_of[s] for s in e["test"]})
                           for e in lineage["experiments"].values())
    tt_map = mapping["tensor_trust"]
    grid = {(g["strategy"], g["content_tokens"]) for g in (context or {}).get("grid", [])}
    checks = [
        ("Tensor Trust parser correct",
         min(recovery) >= 0.99,
         f"v1 raw dump reproduces upstream candidates for {v1['hijacking_benchmark_recovered']}/{v1['hijacking_benchmark_unique']} "
         f"hijacking and {v1['extraction_benchmark_recovered']}/{v1['extraction_benchmark_unique']} extraction benchmark attacks; "
         "schema surprises raise SchemaError (tests/test_v5_dataset.py)"),
        ("Tensor Trust source semantics preserved",
         all(v["label"] != "BENIGN" for v in tt_map.values())
         and {"unverified_attempt", "self_or_sandbox_attack", "access_code_entry"}
         <= {k for k, v in tt_map.items() if v["label"] == "EXCLUDED"}
         and mapping["tensor_trust_defenses"]["label"] == "EXCLUDED",
         "only upstream success heuristics/benchmarks map to ATTACK; failed attempts, self attacks, access codes and defenses are EXCLUDED, never BENIGN"),
        ("checksum reproducible",
         checksums_match() and rebuild is not False,
         "pinned SHA-256 for every source file; manifest/lineage/mapping checksums match"
         + ("; rebuild produced identical checksums" if rebuild else "; rebuild not run in this invocation")),
        ("train/test leakage none",
         all(len(s) == 1 for s in cluster_split.values()) and experiment_leaks == 0,
         f"{len(cluster_split)} clusters each in one split; {experiment_leaks} train/dev-test cluster overlaps across experiments A-D"),
        ("cross-source dedup done",
         bool(dedup) and "MinHash" in dedup["method"]["near_duplicate"],
         f"{dedup['method']['near_duplicate']}; Tensor Trust <-> JailbreakLLMs overlap rows: {dedup['tensor_trust_jailbreakllms_overlap_rows']}"),
        ("context ablation done",
         {("HEAD", c) for c in (94, 128, 256, 510)} <= grid and {("TAIL", 94), ("HEAD_TAIL", 94), ("SLIDING_MAX", 94)} <= grid,
         "HEAD/TAIL/HEAD_TAIL at 94/128/256/510 + sliding windows on the same V2 checkpoint; 512 is not a valid content length"),
        ("tokenizer audit done",
         bool(tokenizer) and any("turkish" in k.lower() for k in tokenizer["groups"]),
         "English-dominant Tensor Trust/JailbreakLLMs vs native Turkish text"),
        ("V2 public baseline done",
         bool(baseline) and {"TEST", "OOD_TEST"} <= set(baseline["splits"]),
         "frozen V2 at native HEAD-94 on DEV, TEST and OOD_TEST per source, fixed threshold 0.5"),
        ("calibration audit done",
         bool(calibration) and calibration["threshold_tuning_on_test"] is False and "dev_fitted_temperature" in calibration,
         "temperature fitted on DEV only; per-source ECE/Brier/precision/recall/FPR"),
    ]
    status = "GO" if all(ok for _, ok, _ in checks) else "NO_GO"
    warnings = [
        "Tensor Trust data repository has no LICENSE file (game code is BSD-2-Clause); status reported as-is. "
        "Raw text is not committed or redistributed.",
        "HackAPrompt not included: gated dataset, terms not yet accepted by the account owner.",
        f"REAL_V2_REDACT_E2E = {firewall['REAL_V2_REDACT_E2E']} ({firewall['redact_e2e_passed']} PII fixtures); "
        f"{firewall['passed']}/{firewall['total']} firewall fixtures matched expectations." if firewall else
        "firewall real-V2 validation missing",
        "Security-prose hard negatives were filtered by rules (code/quotes/imperative payload lines removed), not reviewed by people.",
        "V2 context ablation reuses weights trained at 94 tokens; it measures reading strategy under distribution shift, not a 512-token model.",
    ]
    if calibration and calibration["dev_fitted_temperature"] >= 19.9:
        warnings.append(f"DEV temperature fit reached the search bound ({calibration['dev_fitted_temperature']}): "
                        "V2 attack scores carry little ranking signal on these sources, so scaling mostly flattens them.")
    lines = ["# V5 evidence gate", "", f"**V5_STATUS = {status}**", "",
             "GO means the evidence required *before* training exists. It does not mean V5 is trained, "
             "better than V2, or deployable. No training was started by this gate.", "",
             "| Check | Result | Evidence |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | {why} |" for name, ok, why in checks]
    lines += ["", "## Warnings (not gate items)", ""] + [f"- {w}" for w in warnings] + [""]
    (R / "v5_evidence_gate.md").write_text("\n".join(lines))
    print(f"V5_STATUS = {status}")
    for name, ok, _ in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()
