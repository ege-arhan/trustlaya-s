"""Generate per-task raw/calibrated reliability diagrams as standalone SVG."""

import json
from pathlib import Path

from trustlaya.dataset import read_rows
from trustlaya.inference import Analyzer
from trustlaya.labels import TASKS
from trustlaya.calibration import metrics

ROOT = Path(__file__).resolve().parents[1]


def bins(labels, probabilities):
    result = []
    for index in range(10):
        chosen = [(int(y), float(p)) for y, p in zip(labels, probabilities)
                  if index / 10 <= p < (index + 1) / 10 or
                  (index == 9 and p == 1)]
        if chosen:
            result.append({"bin": index, "n": len(chosen),
                           "predicted": sum(p for _, p in chosen) / len(chosen),
                           "observed": sum(y for y, _ in chosen) / len(chosen)})
    return result


def main():
    rows = read_rows(ROOT / "data/splits/val.jsonl")
    analyzer = Analyzer("onnx", model_dir=ROOT / "models/trustlaya-s-v2",
                        onnx_path=ROOT / "models/exported/v2/trustlaya_s.onnx")
    predictions = [analyzer.analyze(row["text"]) for row in rows]
    report = {}
    scores = {}
    for task in TASKS:
        labels = [row["labels"][task] for row in rows]
        raw = [row["raw_scores"][task] for row in predictions]
        calibrated = [row["calibrated_scores"][task] for row in predictions]
        report[task] = {
            "raw": bins(labels, raw),
            "calibrated": bins(labels, calibrated),
        }
        scores[task] = {"raw": metrics(labels, raw),
                        "calibrated": metrics(labels, calibrated)}
    (ROOT / "reports/reliability_v2.json").write_text(json.dumps(report, indent=2))
    (ROOT / "reports/calibration_v2.json").write_text(json.dumps({
        "scope": "original synthetic validation, 1053 English-labeled rows",
        "selection_note": "Eight v1 temperatures were fitted on this validation set; PII v2 temperature was fitted on separate Turkish privacy development scenarios. These are not independent calibration estimates.",
        "tasks": scores,
        "mean": {kind: {metric: sum(scores[task][kind][metric] for task in TASKS) / len(TASKS)
                        for metric in ("ece", "brier", "nll")}
                 for kind in ("raw", "calibrated")}}, indent=2))
    width, height = 930, 780
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="TrustLaya-S per-task reliability diagrams">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<text x="30" y="30" font-family="sans-serif" font-size="19" font-weight="bold">TrustLaya-S v2 — validation reliability</text>',
             '<text x="30" y="49" font-family="sans-serif" font-size="11" fill="#475569">Synthetic English-labeled validation; raw blue, calibrated orange. Empty bins omitted.</text>']
    for n, task in enumerate(TASKS):
        col, row = n % 3, n // 3
        x, y, side = 52 + col * 305, 95 + row * 225, 150
        parts.append(f'<text x="{x}" y="{y-12}" font-family="sans-serif" font-size="12" font-weight="bold">{task.replace("_", " ")}</text>')
        parts.append(f'<rect x="{x}" y="{y}" width="{side}" height="{side}" fill="white" stroke="#94a3b8"/>')
        parts.append(f'<path d="M{x} {y+side} L{x+side} {y}" stroke="#cbd5e1" stroke-dasharray="4 3"/>')
        for label, color in (("raw", "#2563eb"), ("calibrated", "#ea580c")):
            points = " ".join(f'{x+item["predicted"]*side:.2f},{y+(1-item["observed"])*side:.2f}'
                              for item in report[task][label])
            if points:
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
                for point in points.split():
                    px, py = point.split(",")
                    parts.append(f'<circle cx="{px}" cy="{py}" r="2.5" fill="{color}"/>')
        parts.append(f'<text x="{x}" y="{y+side+16}" font-family="sans-serif" font-size="10">0</text>')
        parts.append(f'<text x="{x+side-6}" y="{y+side+16}" font-family="sans-serif" font-size="10">1</text>')
    parts.append('</svg>')
    (ROOT / "reports/reliability_v2.svg").write_text("\n".join(parts))
    print(f"Wrote reliability diagram for {len(TASKS)} tasks and {len(rows)} validation rows")


if __name__ == "__main__":
    main()
