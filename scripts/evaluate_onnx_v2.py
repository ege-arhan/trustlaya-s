"""Compare versioned FP32/INT8 ONNX outputs with PyTorch on identical inputs."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.special import expit
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from trustlaya.calibration import ece
from trustlaya.dataset import read_rows
from trustlaya.evidence import extract, PII_TYPES, SECRET_TYPES
from trustlaya.labels import TASKS
from trustlaya.model import TrustLaya, BACKBONE
from trustlaya.inference import Analyzer
from trustlaya.utils import normalize

ROOT = Path(__file__).resolve().parents[1]


def main():
    model_dir = ROOT / "models/trustlaya-s-v2"
    exported = ROOT / "models/exported/v2"
    fp32 = exported / "trustlaya_s.onnx"
    int8 = exported / "trustlaya_s_int8.onnx"
    onnx.checker.check_model(str(fp32))
    onnx.checker.check_model(str(int8))
    model = TrustLaya(BACKBONE, pretrained=False)
    model.load(model_dir / "model.safetensors")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sessions = {"fp32": ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]),
                "int8": ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])}
    rows = read_rows(ROOT / "data/splits/test.jsonl")[:256]
    outputs = {name: [] for name in ("torch", "fp32", "int8")}
    for start in range(0, len(rows), 16):
        chunk = rows[start:start + 16]
        tokens = tokenizer([normalize(row["text"]) for row in chunk],
                           return_tensors="np", truncation=True,
                           padding="max_length", max_length=96)
        inputs = {key: value.astype("int64") for key, value in tokens.items()
                  if key in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            result = model(torch.from_numpy(inputs["input_ids"]),
                           torch.from_numpy(inputs["attention_mask"]))
        outputs["torch"].append(tuple(value.numpy() for value in result))
        for name, session in sessions.items():
            outputs[name].append(tuple(session.run(None, inputs)))
    outputs = {name: [np.concatenate([batch[i] for batch in batches]) for i in range(3)]
               for name, batches in outputs.items()}
    temperatures = json.loads((model_dir / "calibration.json").read_text())
    thresholds = json.loads((model_dir / "decision_thresholds.json").read_text())
    report = {"rows": len(rows), "backends": {}, "drift": {}}
    for name, values in outputs.items():
        risks = values[0]
        tasks = {}
        for i, task in enumerate(TASKS):
            probs = expit(risks[:, i] / temperatures[task])
            if task in ("pii", "secret"):
                types = PII_TYPES if task == "pii" else SECRET_TYPES
                for j, row in enumerate(rows):
                    if any(item["type"] in types for item in extract(row["text"])):
                        probs[j] = max(probs[j], .95)
            truth = [row["labels"][task] for row in rows]
            tasks[task] = {"f1": float(f1_score(truth,
                            probs >= thresholds.get(task, .5), zero_division=0)),
                           "ece": ece(truth, probs)}
        report["backends"][name] = {
            "tasks": tasks,
            "macro_f1": sum(row["f1"] for row in tasks.values()) / len(tasks),
            "mean_ece": sum(row["ece"] for row in tasks.values()) / len(tasks),
        }
    for name in ("fp32", "int8"):
        report["drift"][name] = {
            "max_absolute_risk_logit": float(np.max(np.abs(outputs[name][0] - outputs["torch"][0]))),
            "max_absolute_severity_logit": float(np.max(np.abs(outputs[name][1] - outputs["torch"][1]))),
            "max_absolute_action_logit": float(np.max(np.abs(outputs[name][2] - outputs["torch"][2]))),
            "advisory_action_disagreement_rate": float(np.mean(
                np.argmax(outputs[name][2], axis=1) != np.argmax(outputs["torch"][2], axis=1))),
        }
    fp_analyzer = Analyzer("onnx", model_dir=model_dir, onnx_path=fp32)
    int_analyzer = Analyzer("onnx_int8", model_dir=model_dir, onnx_path=int8)
    fp_actions = [fp_analyzer.analyze(row["text"])["action"] for row in rows]
    int_actions = [int_analyzer.analyze(row["text"])["action"] for row in rows]
    report["drift"]["int8"]["final_action_disagreement_rate"] = sum(
        a != b for a, b in zip(fp_actions, int_actions)) / len(rows)
    (ROOT / "reports/onnx_v2_parity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"drift": report["drift"],
                      "summary": {key: {"macro_f1": value["macro_f1"],
                                         "mean_ece": value["mean_ece"]}
                                  for key, value in report["backends"].items()}}, indent=2))


if __name__ == "__main__":
    main()
