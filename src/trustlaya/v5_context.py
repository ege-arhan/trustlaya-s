"""Auditable read strategies; they select context, never assign attack labels."""
from __future__ import annotations

import math
import re

from trustlaya.context_windows import read_windows

STRATEGIES = (
    "FIRST_WINDOW", "LAST_WINDOW", "HEAD_TAIL", "FIXED_SLIDING_WINDOW",
    "MAX_WINDOW_SCORE", "MEAN_WINDOW_SCORE", "TOP_K_RISK_WINDOW",
    "TWO_STAGE_CONTEXT_ROUTER",
)

# Retrieval cues only. No cue is a ground-truth attack decision.
_CUES = re.compile(r"(?i)\b(?:ignore|override|reveal|extract|bypass|system prompt|"
                   r"previous instructions|önceki talimatları|sistem mesajı|kuralları yok say)\b")


def cue_token_positions(text: str, offsets: list[tuple[int, int]]) -> list[int]:
    hits = []
    for match in _CUES.finditer(text):
        position = next((i for i, (start, end) in enumerate(offsets)
                         if start < match.end() and end > match.start()), None)
        if position is not None:
            hits.append(position)
    return hits


def candidate_windows(tokens: list[int], strategy: str, *, cues: list[int] | None = None,
                      capacity: int = 94, overlap: int = 47) -> list[list[int]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    variants = read_windows(tokens, capacity, overlap)
    if strategy == "FIRST_WINDOW":
        return variants["HEAD"]
    if strategy == "LAST_WINDOW":
        return variants["TAIL"]
    if strategy == "HEAD_TAIL":
        return variants["HEAD_TAIL"]
    if strategy != "TWO_STAGE_CONTEXT_ROUTER":
        return variants["SLIDING"]
    starts = {0, max(0, len(tokens) - capacity)}
    # A cheap marker pass selects up to two internal regions. This is retrieval,
    # not classification; both edge regions remain in scope even without cues.
    for position in (cues or [])[:2]:
        starts.add(min(max(0, position - capacity // 2), max(0, len(tokens) - capacity)))
    return [tokens[start:start + capacity] for start in sorted(starts)]


def aggregate_score(scores: list[float], strategy: str) -> float:
    if not scores or strategy not in STRATEGIES:
        raise ValueError("invalid strategy or empty scores")
    if strategy in {"FIRST_WINDOW", "LAST_WINDOW", "HEAD_TAIL"}:
        return float(scores[0])
    if strategy in {"MAX_WINDOW_SCORE", "TWO_STAGE_CONTEXT_ROUTER"}:
        return float(max(scores))
    if strategy == "MEAN_WINDOW_SCORE":
        return float(sum(scores) / len(scores))
    if strategy == "TOP_K_RISK_WINDOW":
        return float(sum(sorted(scores, reverse=True)[:2]) / min(2, len(scores)))
    clipped = [min(max(p, 1e-6), 1 - 1e-6) for p in scores]
    mean_logit = sum(math.log(p / (1 - p)) for p in clipped) / len(clipped)
    return float(1 / (1 + math.exp(-mean_logit)))
