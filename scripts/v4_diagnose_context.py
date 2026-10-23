"""Measure how much of frozen JailbreakLLMs v2/v3 actually reaches the encoder."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from run_external_real import ROOT, LOCAL, jailbreak_samples
from trustlaya.utils import normalize

TOKENIZER = ROOT / "models/trustlaya-s-v2"
MODEL_CONFIG = ROOT / "models/base"
PRED = LOCAL / "v4_length_audit.json"
METRICS = ROOT / "reports/v4_long_context_metrics.json"
REPORT = ROOT / "reports/long_context_diagnostic.md"

# Cues signal possible attack wording, not the semantic truth of a prompt.
CUES = re.compile(
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)"
    r"|(?:reveal|show|print|expose)\s+(?:the\s+|your\s+)?(?:hidden\s+|initial\s+)?(?:system\s+prompt|instructions?|password)"
    r"|(?:you\s+are\s+now|from\s+now\s+on|act\s+as|pretend\s+to\s+be)"
    r"|(?:developer\s+mode|do\s+anything\s+now|no\s+restrictions|without\s+restrictions)"
    r"|(?:bypass|override|disable)\s+(?:the\s+|your\s+)?(?:safety|policy|rules?|instructions?|filter)",
    re.I,
)


def summary(rows):
    n = len(rows)
    tokens = np.asarray([r["original_token_length"] for r in rows], dtype=int)
    chars = np.asarray([r["character_length"] for r in rows], dtype=int)
    count = lambda condition: int(sum(condition(r) for r in rows))
    return {"n": n, "characters": {"median": float(np.median(chars)), "p90": float(np.percentile(chars, 90)),
                                      "p95": float(np.percentile(chars, 95)), "p99": float(np.percentile(chars, 99))},
            "tokens": {"median": float(np.median(tokens)), "p90": float(np.percentile(tokens, 90)),
                       "p95": float(np.percentile(tokens, 95)), "p99": float(np.percentile(tokens, 99)),
                       "maximum": int(tokens.max())},
            "over_current_94": count(lambda r: r["truncated_at_94"]),
            "over_model_capacity_510": count(lambda r: r["over_model_capacity"]),
            "cue_anywhere": count(lambda r: r["cue_count"] > 0),
            "cue_only_after_94": count(lambda r: r["cue_count"] > 0 and r["first_cue_token"] >= 94),
            "cue_only_after_510": count(lambda r: r["cue_count"] > 0 and r["first_cue_token"] >= 510),
            "cue_after_94_any": count(lambda r: r["cue_after_94_any"]),
            "cue_after_510_any": count(lambda r: r["cue_after_510_any"]),
            "mean_retained_ratio_at_94": float(np.mean([r["retained_ratio_94"] for r in rows]))}


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    config = AutoConfig.from_pretrained(MODEL_CONFIG)
    assert tok.model_max_length == config.max_position_embeddings == 512
    special = tok.num_special_tokens_to_add(False)
    assert special == 2
    capacity = config.max_position_embeddings - special
    assert capacity == 510
    frozen = {r["sample_id"]: r for r in json.loads((LOCAL / "v2_frozen_jailbreak_predictions.json").read_text())}
    clean = {r["sample_id"] for r in json.loads((LOCAL / "v3_jailbreak_predictions.json").read_text())}
    samples = [r for r in jailbreak_samples()[0] if r["sample_id"] in frozen]
    assert len(samples) == 5888
    rows = []
    for i, sample in enumerate(samples):
        content = normalize(sample["text"])
        offsets = tok(content, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        length = len(offsets)
        cue_positions = []
        for match in CUES.finditer(content):
            # First token overlapping the cue. Most cues are plain text; missing
            # offset alignment is recorded conservatively as no located cue.
            token = next((j for j, (a, b) in enumerate(offsets) if a < match.end() and b > match.start()), None)
            if token is not None: cue_positions.append(token)
        row = {"sample_id": sample["sample_id"], "text_sha256": hashlib.sha256(content.encode()).hexdigest(),
               "gold": sample["gold"], "source": sample["source"], "clean_v3_cohort": sample["sample_id"] in clean,
               "character_length": len(sample["text"]), "original_token_length": length,
               "retained_token_length_94": min(length, 94), "truncated_at_94": length > 94,
               "retained_ratio_94": min(length, 94) / max(1, length),
               "retained_token_length_510": min(length, capacity), "over_model_capacity": length > capacity,
               "cue_count": len(cue_positions), "first_cue_token": min(cue_positions) if cue_positions else None,
               "cue_after_94_any": any(x >= 94 for x in cue_positions),
               "cue_after_510_any": any(x >= capacity for x in cue_positions)}
        rows.append(row)
        if i and i % 1000 == 0: print("tokenized", i, "/", len(samples), flush=True)
    PRED.write_text(json.dumps(rows, indent=2) + "\n")
    cohorts = {"full": rows, "clean": [r for r in rows if r["clean_v3_cohort"]]}
    metrics = {name: {"all": summary(items), "attack": summary([r for r in items if r["gold"] == 1]),
                      "regular": summary([r for r in items if r["gold"] == 0])}
               for name, items in cohorts.items()}
    metrics["settings"] = {"tokenizer_model_max_length_total": tok.model_max_length,
                            "model_position_embeddings_total": config.max_position_embeddings,
                            "special_tokens": special, "single_pass_content_capacity": capacity,
                            "actual_evaluation_max_length_total": 96,
                            "actual_evaluation_content_capacity": 94,
                            "cue_regex": CUES.pattern,
                            "v2_tokenizer_sha256": hashlib.sha256((TOKENIZER / "tokenizer.json").read_bytes()).hexdigest()}
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n")
    clean_attack, clean_regular = metrics["clean"]["attack"], metrics["clean"]["regular"]
    pct = lambda value, denominator: f"{100 * value / denominator:.1f}%"
    report = f"""# Long-context diagnostic before v4 training

This diagnostic used the frozen JailbreakLLMs community rows (full N=5,888; clean paired N=5,761), the exact v2/v3 tokenizer, and the existing `normalize` function. No model was trained or threshold chosen. The [machine-readable metrics](v4_long_context_metrics.json) and gitignored per-row audit `benchmarks/external/predictions/v4_length_audit.json` contain hashes, lengths and cue positions, **not raw prompt text**.

## Actual limits

The BERT encoder has `max_position_embeddings=512`, and the tokenizer declares a **512 total-token** maximum. Two special tokens leave **510 content tokens** for a valid single pass. The current v2 `Analyzer.analyze()` and v3 head extraction both set `max_length=96`, so they expose only the **first 94 content tokens**. This 96-token application choice, not the architecture, is the current external-evaluation truncation boundary.

| Clean test class | N | Median chars | Median tokens | p90 tokens | p95 tokens | p99 tokens | >94 tokens/currently truncated | >510/model single-pass capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attack | {clean_attack['n']:,} | {clean_attack['characters']['median']:.0f} | {clean_attack['tokens']['median']:.0f} | {clean_attack['tokens']['p90']:.0f} | {clean_attack['tokens']['p95']:.0f} | {clean_attack['tokens']['p99']:.0f} | {clean_attack['over_current_94']:,} ({pct(clean_attack['over_current_94'], clean_attack['n'])}) | {clean_attack['over_model_capacity_510']:,} ({pct(clean_attack['over_model_capacity_510'], clean_attack['n'])}) |
| Regular | {clean_regular['n']:,} | {clean_regular['characters']['median']:.0f} | {clean_regular['tokens']['median']:.0f} | {clean_regular['tokens']['p90']:.0f} | {clean_regular['tokens']['p95']:.0f} | {clean_regular['tokens']['p99']:.0f} | {clean_regular['over_current_94']:,} ({pct(clean_regular['over_current_94'], clean_regular['n'])}) | {clean_regular['over_model_capacity_510']:,} ({pct(clean_regular['over_model_capacity_510'], clean_regular['n'])}) |

In the clean positive set, {clean_attack['cue_anywhere']} prompts contain at least one regex cue for override, system-prompt disclosure, role change or policy bypass. In **{clean_attack['cue_only_after_94']}** of these, the *first* matched cue begins at or beyond token 94, so the current head-only evaluation cannot see that matched phrase. {clean_attack['cue_after_94_any']} have at least one matched cue beyond 94. At the physical 510-token limit, {clean_attack['cue_only_after_510']} have the first cue beyond a valid single pass. The regex is a lexical heuristic: it misses indirect/obfuscated attacks and can match quoted text. These counts are **not gold attack-span recall**.

**Answer:** the current classifier sees at most the first 94 content tokens per prompt. It truncates {pct(clean_attack['over_current_94'], clean_attack['n'])} of attacks and {pct(clean_regular['over_current_94'], clean_regular['n'])} of regular prompts. A 512-token single pass would still truncate {pct(clean_attack['over_model_capacity_510'], clean_attack['n'])} of attacks. The next experiment must compare head, tail and windowed inference on development data before changing weights.
"""
    REPORT.write_text(report)
    print("Saved", REPORT)


if __name__ == "__main__": main()
