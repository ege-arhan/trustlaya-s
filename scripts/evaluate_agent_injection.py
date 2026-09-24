"""Evaluate text detection on a separate agentic injection benchmark.

This measures payload classification, not whether an agent executes an attack.
"""

import hashlib
import json
import gc
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import onnxruntime as ort
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score

from trustlaya.inference import Analyzer
from trustlaya.utils import normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "ppradyoth/AgentInjectionBench"
SOURCE_REVISION = "ef230359966c4d87b1c925aa56b8bf6e3f6ffed3"
FILE = "data/agent_injection_bench.jsonl"


def metrics(rows, threshold):
    truth = [row["label"] for row in rows]
    prediction = [int(row["score"] >= threshold) for row in rows]
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    return {"n": len(rows), "positive": int(sum(truth)), "threshold": threshold,
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "mcc": float(matthews_corrcoef(truth, prediction)),
            "false_positive_rate": float(fp / max(1, fp + tn)),
            "false_negative_rate": float(fn / max(1, fn + tp)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def content(row, include_definitions=False):
    # Only lower-trust tool content is scored. User/assistant turns are context,
    # not the attack-bearing surface in this published benchmark.
    parts = [str(turn.get("content") or "") for turn in row["conversation"]
             if turn.get("role") == "tool_result"]
    if include_definitions:
        parts += [str(tool.get("description") or "") for tool in row.get("tools_available", [])
                  if isinstance(tool, dict)]
    return "\n".join(parts)


def sliding_score(analyzer, text):
    text = normalize(text)
    offsets = analyzer.tokenizer(text, add_special_tokens=False,
                                 return_offsets_mapping=True)["offset_mapping"]
    if len(offsets) <= 94:
        return analyzer.analyze(text)["prompt_injection"], 1
    starts = list(range(0, len(offsets) - 94, 70))
    starts.append(len(offsets) - 94)
    scores = [analyzer.analyze(text[offsets[start][0]:offsets[start + 93][1]])["prompt_injection"]
              for start in sorted(set(starts))]
    return max(scores), len(scores)


def main():
    ort.disable_telemetry_events()
    path = Path(hf_hub_download(SOURCE, FILE, repo_type="dataset", revision=SOURCE_REVISION))
    source_revision = HfApi().dataset_info(SOURCE, revision=SOURCE_REVISION).sha
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = [json.loads(line) for line in path.open()]
    analyzer = Analyzer("onnx", model_dir=ROOT / "models/trustlaya-s-v2",
                        onnx_path=ROOT / "models/exported/v2/trustlaya_s.onnx")
    views = {}
    for name, definitions, sliding in (("tool_results", False, False),
                                       ("tool_results_plus_definitions", True, False),
                                       ("tool_results_sliding_max", False, True)):
        evaluated = []
        for row in raw:
            text = content(row, definitions)
            if not text:
                score, windows = 0.0, 0
            elif sliding:
                score, windows = sliding_score(analyzer, text)
            else:
                score, windows = analyzer.analyze(text)["prompt_injection"], 1
            evaluated.append({"label": int(row["ground_truth"] == "unsafe"),
                              "score": score, "category": row["attack_category"],
                              "bypass": row["defense_bypass"], "surface": row["injection_surface"],
                              "characters": len(text), "tokens": len(analyzer.tokenizer(
                                  normalize(text), add_special_tokens=False)["input_ids"]),
                              "windows": windows})
        result = {"at_0_5": metrics(evaluated, 0.5),
                  "at_policy_0_7": metrics(evaluated, 0.7),
                  "empty_payloads": sum(not row["characters"] for row in evaluated),
                  "truncated_at_96_tokens": sum(row["tokens"] > 94 for row in evaluated),
                  "total_windows": sum(row["windows"] for row in evaluated)}
        for field in ("category", "bypass", "surface"):
            groups = defaultdict(list)
            for row in evaluated:
                groups[row[field]].append(row)
            result[field] = {key: metrics(items, 0.5) for key, items in groups.items()
                             if len({item["label"] for item in items}) == 2}
            result[field + "_positive_recall"] = {
                key: float(sum(item["score"] >= 0.5 for item in items if item["label"]) /
                           max(1, sum(item["label"] for item in items)))
                for key, items in groups.items() if any(item["label"] for item in items)}
        views[name] = result
    report = {"source": SOURCE, "source_revision": source_revision,
              "source_sha256": source_hash, "license": "Apache-2.0",
              "source_counts": dict(Counter(row["ground_truth"] for row in raw)),
              "scope": "Published benchmark tool_result text; second view appends advertised tool descriptions. Third view uses parameter-free overlapping 94-token windows and maximum score. This is prompt-injection text classification, NOT agent attack success rate. Some cases are multi-turn; default model truncates concatenated tool results to 96 tokens.",
              "selection_note": "No training, calibration or threshold selection on this source. Its results are now inspected and it is no longer fresh for future tuning.",
              "views": views}
    target = ROOT / "reports/agent_injection_independent.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    del analyzer
    gc.collect()
    print(json.dumps({name: value["at_0_5"] for name, value in views.items()}, indent=2))


if __name__ == "__main__":
    main()
