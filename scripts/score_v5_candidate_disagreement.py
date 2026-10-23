"""Private sampling signal only: frozen heads on first 94 content tokens."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from run_v3_external_eval import apply_calibration, load_model
from run_v5_context_diagnostic import score_windows
from trustlaya.utils import normalize
from trustlaya.v5_benchmark import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "benchmarks/v5/private"


def main() -> None:
    metadata = read_jsonl(ROOT / "benchmarks/v5/dataset_manifest.jsonl")
    queue = read_jsonl(PRIVATE / "annotation_queue.jsonl")
    if [r["sample_id"] for r in metadata] != [r["sample_id"] for r in queue]:
        raise ValueError("manifest/queue mismatch")
    model, tokenizer, _, v3_head, device = load_model()
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    v4_head = torch.nn.Linear(model.encoder.config.hidden_size, 1)
    v4_head.load_state_dict(load_file(ROOT / "models/trustlaya-s-v4-research/attack_intent_head.safetensors"))
    v4_head.eval()
    windows = [tokenizer.backend_tokenizer.encode(normalize(row["text"]), add_special_tokens=False).ids[:94]
               for row in queue]
    raw = score_windows(model, {"v3": v3_head, "v4": v4_head}, device, tokenizer, windows)
    v3_fit = json.loads((ROOT / "models/trustlaya-s-v3/calibration.json").read_text())["attack"]
    v4_protocol = json.loads((ROOT / "models/trustlaya-s-v4-research/operating_point.json").read_text())
    v3 = apply_calibration(raw["v3"], v3_fit)
    v4 = apply_calibration(raw["v4"], v4_protocol["calibration"])
    output = {}
    for index, row in enumerate(metadata):
        decisions = {"v2": bool(raw["v2"][index] >= .5),
                     "v3": bool(v3[index] >= .3),
                     "v4": bool(v4[index] >= v4_protocol["threshold"])}
        output[row["sample_id"]] = {"decisions": decisions,
                                    "disagreement": len(set(decisions.values())) > 1}
    (PRIVATE / "candidate_disagreement.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"n": len(output), "disagreement_n": sum(x["disagreement"] for x in output.values()),
                      "note": "sampling signal only; no human labels or V5 performance inferred"}))


if __name__ == "__main__":
    main()
