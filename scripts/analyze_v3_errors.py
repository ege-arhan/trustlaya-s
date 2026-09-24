"""Deterministic private error triage. Categories are heuristic, not human gold."""
from __future__ import annotations

import json
import random
import re
from collections import Counter

from run_external_real import ROOT, LOCAL, jailbreak_samples, tab_samples
from transformers import AutoTokenizer

SEED = 20260925


def categories(text, task, kind):
    t = text.casefold()
    out = []
    if "JAILBREAK" in task:
        if len(text) <= 40: out.append("very_short")
        if len(text) >= 1000: out.append("long_context")
        if re.search(r"ignore.{0,40}(previous|prior|instruction)|forget.{0,30}(rule|instruction)|override.{0,30}(system|instruction)", t): out.append("instruction_override")
        if re.search(r"you are now|act as|pretend (to be|you are)|from now on|roleplay|persona", t): out.append("role_hijack_or_roleplay")
        if re.search(r"system prompt|developer message|initial instruction|hidden instruction", t): out.append("system_prompt_reference")
        if re.search(r"password|api key|secret|token|credential", t): out.append("secret_reference")
        if re.search(r"```|import |function\s*\(|<script|base64|rot13|decode|encode", t): out.append("code_or_encoding")
        if re.search(r"jailbreak|prompt injection|security research|safety policy", t): out.append("security_vocabulary")
        if kind == "FP" and re.search(r"ignore.{0,40}(previous|prior|instruction)|you are now|do anything now", t): out.append("possible_gold_label_noise")
    else:
        if re.search(r"\b(?:mr|mrs|ms|dr|sir)\b", t): out.append("titled_person")
        if re.search(r"\b\d{3,7}/\d{2}\b", t): out.append("application_code")
        if re.search(r"v\.\s+[a-z]|court|chamber|government", t): out.append("legal_context")
        if len(text) >= 700: out.append("long_window")
    return out or ["other_unclassified"]


def sampled(rows, task, kind):
    selected = [r for r in rows if r["kind"] == kind]
    random.Random(SEED).shuffle(selected)
    return selected[:100]


def main():
    tok = AutoTokenizer.from_pretrained(ROOT / "models/trustlaya-s-v3")
    tab = {r["sample_id"]: r for r in tab_samples(tok, "test")[0]}
    jail = {r["sample_id"]: r for r in jailbreak_samples()[0]}
    result = {}
    for task, path, source, threshold in (
        ("PII", "v3_tab_predictions.json", tab, json.loads((ROOT / "models/trustlaya-s-v3/decision_thresholds.json").read_text())["pii"]),
        ("JAILBREAK", "v3_jailbreak_predictions.json", jail, json.loads((ROOT / "models/trustlaya-s-v3/decision_thresholds.json").read_text())["attack"]),
        ("V2_JAILBREAK", "v2_frozen_jailbreak_predictions.json", jail, .5),
    ):
        rows = []
        for r in json.loads((LOCAL / path).read_text()):
            score = r.get("calibrated_score", r.get("raw_score")) if task != "V2_JAILBREAK" else r["raw_score"]
            pred = int(score >= threshold); gold = r["gold"]
            if pred == gold: continue
            kind = "FP" if pred else "FN"
            rows.append({"sample_id": r["sample_id"], "kind": kind, "score": float(score),
                         "text": source[r["sample_id"]]["text"]})
        result[task] = {}
        for kind in ("FP", "FN"):
            selection = sampled(rows, task, kind)
            counts = Counter(c for r in selection for c in categories(r["text"], task, kind))
            result[task][kind] = {"population": sum(r["kind"] == kind for r in rows),
                                  "inspected": len(selection), "heuristic_category_counts": dict(counts),
                                  "sample_ids": [r["sample_id"] for r in selection]}
    (ROOT / "reports/v3_error_triage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
