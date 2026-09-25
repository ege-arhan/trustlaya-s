"""Reliability diagram from frozen v4 external predictions (descriptive only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROWS=json.loads((ROOT/"benchmarks/external/predictions/v4_jailbreak_final_predictions.json").read_text())


def main():
    gold=np.asarray([r["gold"] for r in ROWS])
    width=640;height=520;left=80;top=55;size=390
    def point(x,y):return f"{left+x*size:.1f},{top+(1-y)*size:.1f}"
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="80" y="27" font-family="sans-serif" font-size="18">JailbreakLLMs v4 reliability, N=5,761</text>',
           f'<line x1="{left}" y1="{top+size}" x2="{left+size}" y2="{top}" stroke="#555" stroke-dasharray="5,5"/>',
           f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+size}" stroke="black"/>',
           f'<line x1="{left}" y1="{top+size}" x2="{left+size}" y2="{top+size}" stroke="black"/>',
           '<text x="215" y="497" font-family="sans-serif" font-size="14">Mean predicted attack score</text>',
           '<text x="18" y="270" transform="rotate(-90 18 270)" font-family="sans-serif" font-size="14">Observed attack fraction</text>']
    for field,name,color in (("raw_score","raw","#c45a3a"),("calibrated_score","DEV-calibrated","#256f8d")):
        p=np.asarray([r[field] for r in ROWS]); centers=[];freq=[]
        for i in range(10):
            mask=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1)
            if mask.any():centers.append(float(p[mask].mean()));freq.append(float(gold[mask].mean()))
        coords=" ".join(point(x,y) for x,y in zip(centers,freq))
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in zip(centers,freq):
            px,py=point(x,y).split(",")
            lines.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/>')
        lines.append(f'<text x="500" y="{100 if field=="raw_score" else 123}" fill="{color}" font-family="sans-serif" font-size="14">{name}</text>')
    output=ROOT/"reports/v4_reliability.svg"
    output.write_text("\n".join(lines+["</svg>"])+"\n")
    print(output)


if __name__=="__main__":main()
