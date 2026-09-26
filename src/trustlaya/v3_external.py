"""External-data v3 heads. V2 encoder and all original heads stay frozen."""
from __future__ import annotations

import hashlib
import re

import numpy as np

BIO = ("O", "B-PERSON", "I-PERSON", "B-CODE", "I-CODE")


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def token_labels(offsets: list[tuple[int, int]], spans: list[tuple[int, int, str]]) -> list[int]:
    """Project actual TAB DIRECT PERSON/CODE spans onto token offsets."""
    labels = [0] * len(offsets)
    for start, end, kind in dict.fromkeys(map(tuple, spans)):
        if kind not in ("PERSON", "CODE"):
            raise ValueError(f"Unsupported TAB span: {kind}")
        indexes = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
        if not indexes:
            continue
        base = 1 if kind == "PERSON" else 3
        for j, i in enumerate(indexes):
            if labels[i] != 0:
                raise ValueError("Overlapping DIRECT labels")
            labels[i] = base + int(j > 0)
    return labels


def decode_spans(labels: list[int], offsets: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    spans = []
    current = None
    for label, (start, end) in zip(labels, offsets):
        kind = "PERSON" if label in (1, 2) else "CODE" if label in (3, 4) else None
        if kind is None or end <= start:
            if current is not None:
                spans.append(tuple(current)); current = None
            continue
        begin = label in (1, 3)
        # WordPiece sometimes emits B for the next subtoken of the same entity.
        if current is not None and current[2] == kind and (not begin or start == current[1]):
            current[1] = end
        else:
            if current is not None:
                spans.append(tuple(current))
            current = [start, end, kind]
    if current is not None:
        spans.append(tuple(current))
    return spans


def canonicalize_spans(text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Trim token punctuation and recover TAB's legal application-number offsets."""
    result = []
    for start, end, kind in spans:
        surface = text[start:end]
        if kind == "CODE":
            codes = list(re.finditer(r"\b\d{3,7}/\d{2}\b", surface))
            if codes:
                result.extend((start + m.start(), start + m.end(), kind) for m in codes)
                continue
        a, b = start, end
        while a < b and text[a] in " \t\n()[]{}\"“”": a += 1
        while b > a and text[b - 1] in " \t\n()[]{}\"“”.,;:": b -= 1
        if b > a: result.append((a, b, kind))
    return result


def binary_metrics(gold: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    gold = np.asarray(gold, dtype=np.int8); scores = np.asarray(scores, dtype=float)
    pred = scores >= threshold
    tp = int(((gold == 1) & pred).sum()); fp = int(((gold == 0) & pred).sum())
    tn = int(((gold == 0) & ~pred).sum()); fn = int(((gold == 1) & ~pred).sum())
    div = lambda x, y: x / y if y else None
    return {"n": len(gold), "positive": int(gold.sum()), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "fpr": div(fp, fp + tn),
            "fnr": div(fn, fn + tp),
            "roc_auc": float(roc_auc_score(gold, scores)) if len(set(gold)) == 2 else None,
            "pr_auc": float(average_precision_score(gold, scores)) if len(set(gold)) == 2 else None}


def calibration_metrics(gold: np.ndarray, scores: np.ndarray) -> dict:
    gold = np.asarray(gold); scores = np.asarray(scores, dtype=float)
    bins = np.minimum((scores * 10).astype(int), 9)
    ece = sum(float((bins == b).mean()) * abs(float(gold[bins == b].mean()) - float(scores[bins == b].mean()))
              for b in range(10) if (bins == b).any())
    p = np.clip(scores, 1e-7, 1 - 1e-7)
    return {"ece": ece, "brier": float(np.mean((scores - gold) ** 2)),
            "nll": float(-np.mean(gold * np.log(p) + (1 - gold) * np.log(1 - p)))}


def span_metrics(gold_rows: list[list[tuple[int, int, str]]], pred_rows: list[list[tuple[int, int, str]]]) -> dict:
    counts = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in ("PERSON", "CODE", "ALL")}
    for gold, pred in zip(gold_rows, pred_rows, strict=True):
        g, p = set(map(tuple, gold)), set(map(tuple, pred))
        for kind in counts:
            gg = {x for x in g if kind == "ALL" or x[2] == kind}
            pp = {x for x in p if kind == "ALL" or x[2] == kind}
            counts[kind]["tp"] += len(gg & pp)
            counts[kind]["fp"] += len(pp - gg)
            counts[kind]["fn"] += len(gg - pp)
    for row in counts.values():
        tp, fp, fn = (row[k] for k in ("tp", "fp", "fn"))
        row.update(precision=tp / (tp + fp) if tp + fp else None,
                   recall=tp / (tp + fn) if tp + fn else None,
                   f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    return counts
