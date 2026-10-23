"""Private error IDs with conservative heuristic tags; no raw prompt export."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from run_external_real import jailbreak_samples

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/"benchmarks/external/predictions"


def tags(text,length,kind):
    t=text.casefold(); out=[]
    patterns={"security_discussion":r"what is (?:prompt injection|a jailbreak)|explain (?:prompt injection|jailbreak)|security (?:research|paper|report)",
              "quoted_attack":r"[\"'“‘].{0,100}(?:ignore previous|reveal (?:the |your )?system prompt)",
              "roleplay":r"roleplay|act as|pretend to be|fictional|simulate",
              "explicit_override":r"ignore (?:all |the |your )?(?:previous|prior) instructions|disregard (?:all |the |your )?(?:previous|prior) instructions",
              "prompt_extraction":r"(?:reveal|show|print|expose).{0,30}system prompt",
              "obfuscation_hint":r"base64|rot13|cipher|decode|encoded"}
    for label,pattern in patterns.items():
        if re.search(pattern,t):out.append(label)
    if length>510:out.append("over_510_tokens")
    elif length>94:out.append("over_94_tokens")
    if not out:out.append("unclassified_by_heuristic")
    return out


def main():
    predictions=json.loads((LOCAL/"v4_jailbreak_final_predictions.json").read_text())
    text={r["sample_id"]:r["text"] for r in jailbreak_samples()[0]}
    selected={"FP":[],"FN":[]}
    for row in predictions:
        kind="FP" if row["gold"]==0 and row["predicted_label"]==1 else "FN" if row["gold"]==1 and row["predicted_label"]==0 else None
        if kind: selected[kind].append(row)
    selected["FP"].sort(key=lambda r:-r["calibrated_score"])
    selected["FN"].sort(key=lambda r:r["calibrated_score"])
    summary={"method":"non-exclusive regex heuristic tags on first 100 FP and first 100 FN; not human annotation",
             "total_FP":len(selected["FP"]),"total_FN":len(selected["FN"]),"sampled":{},"tag_counts":{}}
    private={}
    for kind in ("FP","FN"):
        chosen=selected[kind][:100]
        items=[{"sample_id":r["sample_id"],"text_hash":r["text_hash"],
                "score":r["calibrated_score"],"token_length":r["token_length"],
                "tags":tags(text[r["sample_id"]],r["token_length"],kind)} for r in chosen]
        private[kind]=items;summary["sampled"][kind]=len(items)
        summary["tag_counts"][kind]=dict(Counter(t for r in items for t in r["tags"]))
    (LOCAL/"v4_error_ids_private.json").write_text(json.dumps(private,indent=2)+"\n")
    (ROOT/"reports/v4_error_counts.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":main()
