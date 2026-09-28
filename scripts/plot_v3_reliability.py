"""Dependency-free SVG reliability plots from local held-out row predictions."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "benchmarks/external/predictions"


def bins(rows, key):
    out = []
    for i in range(10):
        part = [r for r in rows if min(9, int(r[key] * 10)) == i]
        out.append({"bin": i, "n": len(part), "confidence": sum(r[key] for r in part) / len(part) if part else None,
                    "frequency": sum(r["gold"] for r in part) / len(part) if part else None})
    return out


def main():
    data = {}
    for task, name in (("PII", "tab"), ("JAILBREAK", "jailbreak")):
        rows = json.loads((LOCAL / f"v3_{name}_predictions.json").read_text())
        data[task] = {"raw": bins(rows, "raw_score"), "calibrated": bins(rows, "calibrated_score")}
    (ROOT / "reports/v3_reliability_bins.json").write_text(json.dumps(data, indent=2) + "\n")
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="480" viewBox="0 0 900 480">',
             '<rect width="900" height="480" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#243040} .axis{stroke:#64748b;stroke-width:1}.ideal{stroke:#b7c4d0;stroke-dasharray:4 4}.raw{stroke:#d97706;fill:none;stroke-width:2}.cal{stroke:#2563eb;fill:none;stroke-width:2}</style>']
    for index, task in enumerate(data):
        x0 = 55 + 445 * index; y0 = 410; side = 340
        parts += [f'<text x="{x0}" y="37" font-size="20">{html.escape(task)}</text>',
                  f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0+side}" y2="{y0}"/>',
                  f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0-side}"/>',
                  f'<line class="ideal" x1="{x0}" y1="{y0}" x2="{x0+side}" y2="{y0-side}"/>',
                  f'<text x="{x0+90}" y="450" font-size="14">Mean score (10 bins)</text>',
                  f'<text x="{x0}" y="65" font-size="13">Observed positive rate</text>']
        for label, color in (("raw", "raw"), ("calibrated", "cal")):
            points = " ".join(f'{x0+round(b["confidence"]*side,1)},{y0-round(b["frequency"]*side,1)}'
                              for b in data[task][label] if b["n"])
            parts.append(f'<polyline class="{color}" points="{points}"/>')
        parts += [f'<line class="raw" x1="{x0+190}" y1="37" x2="{x0+218}" y2="37"/>',
                  f'<text x="{x0+224}" y="42" font-size="13">raw</text>',
                  f'<line class="cal" x1="{x0+275}" y1="37" x2="{x0+303}" y2="37"/>',
                  f'<text x="{x0+309}" y="42" font-size="13">calibrated</text>']
    parts.append('</svg>')
    (ROOT / "reports/v3_reliability.svg").write_text("\n".join(parts) + "\n")


if __name__ == "__main__": main()
