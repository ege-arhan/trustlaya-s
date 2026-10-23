"""V5 evidence gate: tokenizer audit, Tensor Trust lengths, V2 public baseline,
context ablation, calibration audit and source generalization. No training.

Uses the frozen V2 checkpoint (models/trustlaya-s-v2/model.safetensors) in
PyTorch because the exported ONNX graph is fixed at 96 tokens. Attack score is
the raw sigmoid of the prompt_injection head. The decision threshold 0.5 was
fixed before any of these data were scored (same as the earlier external
evaluation) and is never tuned on TEST. Temperature is fitted on DEV only.

Needs data/v5/all_rows.jsonl from scripts/build_v5_training_set.py.
Run: .venv/bin/python scripts/evaluate_v5_evidence.py
"""

import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v5_training_set import CUE, EXPERIMENTS, POOL  # noqa: E402
from fetch_tensor_trust import ROOT, sha256  # noqa: E402
from trustlaya.labels import TASKS  # noqa: E402
from trustlaya.model import BACKBONE, TrustLaya  # noqa: E402
from trustlaya.utils import device, normalize  # noqa: E402

MODEL_DIR = ROOT / "models/trustlaya-s-v2"
REPORTS = ROOT / "reports"
THRESHOLD = 0.5
HEAD = TASKS.index("prompt_injection")
CONTEXTS = (94, 128, 256, 510)
MAX_POSITIONS = 512  # BERT positions incl. [CLS] and [SEP]
TURKISH = ("Overfit-GM/turkish-toxic-language", "5723921eb712daaf200735ec40eef4f538faa88c",
           "turkish_toxic_language.csv")


def load_rows():
    return [json.loads(line) for line in open(ROOT / "data/v5/all_rows.jsonl")]


def wilson(k, n):
    if not n:
        return None
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def metrics(y, p, threshold=THRESHOLD):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    div = lambda a, b: round(a / b, 4) if b else None
    out = {"n": len(y), "positive": int(y.sum()), "negative": int((y == 0).sum()),
           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
           "recall_95ci": wilson(tp, tp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
           "fpr": div(fp, fp + tn), "fpr_95ci": wilson(fp, fp + tn), "fnr": div(fn, fn + tp),
           "brier": round(float(np.mean((p - y) ** 2)), 4)}
    bins = np.minimum((p * 10).astype(int), 9)
    out["ece"] = round(float(sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
                                 for b in range(10) if (bins == b).any())), 4)
    out["pr_auc"] = round(float(average_precision_score(y, p)), 4) if 0 < y.sum() < len(y) else None
    if not out["negative"]:
        out["note"] = "attack-only set: precision/FPR/PR-AUC undefined"
    return out


class Scorer:
    """Raw prompt_injection probability for token windows, cached per window."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.device = device()
        self.model = TrustLaya(BACKBONE, pretrained=False)
        self.model.load(MODEL_DIR / "model.safetensors")
        self.model.to(self.device).eval()
        self.cache = {}
        self.cls, self.sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        self.calls = 0

    def ids(self, text):
        return self.tokenizer(normalize(text), add_special_tokens=False, verbose=False)["input_ids"]

    def score(self, windows):
        todo = sorted({tuple(w) for w in windows} - self.cache.keys(), key=len)
        for start in range(0, len(todo), 32):
            batch = todo[start:start + 32]
            width = max(len(w) for w in batch) + 2
            ids = torch.zeros(len(batch), width, dtype=torch.long)
            mask = torch.zeros(len(batch), width, dtype=torch.long)
            for i, w in enumerate(batch):
                seq = [self.cls, *w, self.sep]
                ids[i, :len(seq)] = torch.tensor(seq)
                mask[i, :len(seq)] = 1
            with torch.inference_mode():
                logits = self.model(ids.to(self.device), mask.to(self.device))[0][:, HEAD]
            for w, value in zip(batch, torch.sigmoid(logits).float().cpu().tolist()):
                self.cache[w] = value
            self.calls += len(batch)
        return [self.cache[tuple(w)] for w in windows]


def windows(tokens, strategy, capacity):
    if len(tokens) <= capacity:
        return [tokens]
    if strategy == "HEAD":
        return [tokens[:capacity]]
    if strategy == "TAIL":
        return [tokens[-capacity:]]
    if strategy == "HEAD_TAIL":
        left = capacity // 2
        return [tokens[:left] + tokens[-(capacity - left):]]
    if strategy == "SLIDING_MAX":
        stride = capacity // 2
        starts = list(range(0, len(tokens) - capacity + 1, stride))
        if starts[-1] != len(tokens) - capacity:
            starts.append(len(tokens) - capacity)
        return [tokens[s:s + capacity] for s in starts]
    raise ValueError(strategy)


def evaluate_strategy(scorer, items, strategy, capacity):
    all_windows, owners = [], []
    for index, item in enumerate(items):
        for w in windows(item["tokens"], strategy, capacity):
            all_windows.append(w)
            owners.append(index)
    started = time.perf_counter()
    scores = scorer.score(all_windows)
    elapsed = time.perf_counter() - started
    best = [0.0] * len(items)
    for owner, value in zip(owners, scores):
        best[owner] = max(best[owner], value)
    return best, {"model_windows": len(all_windows), "windows_per_input": round(len(all_windows) / len(items), 3),
                  "uncached_seconds": round(elapsed, 2)}


def attack_items(rows):
    return [r for r in rows if r["status"] == "included" and r["label"] in ("ATTACK", "BENIGN")]


def tokenizer_audit(scorer, rows):
    csv_path = next((Path.home() / ".cache/huggingface/hub/datasets--Overfit-GM--turkish-toxic-language/snapshots"
                     / TURKISH[1]).glob(TURKISH[2]), None)
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "included" and r["source"] in ("tensor_trust", "jailbreakllms"):
            groups[f"{r['source']} (English-dominant)"].append(r["text"])
    turkish_note = None
    if csv_path and csv_path.exists():
        frame = pd.read_csv(csv_path)
        native = frame[frame["source"] != "jigsaw"]["text"].dropna().tolist()  # jigsaw rows are translations
        groups["turkish_toxic_language native Turkish (non-Jigsaw)"] = random.Random(42).sample(native, 3000)
        turkish_note = {"dataset": TURKISH[0], "revision": TURKISH[1], "file_sha256": sha256(csv_path),
                        "license": "Apache-2.0", "rows_used": 3000,
                        "selection": "non-Jigsaw rows (Jigsaw rows are machine translations), seed 42"}
    groups["project Turkish benchmark (researcher-authored synthetic)"] = [
        json.loads(line)["text"] for line in open(ROOT / "data/benchmarks/trustlaya_tr_benchmark.jsonl")
        if json.loads(line).get("language") == "tr"]
    out = {}
    for name, texts in groups.items():
        words = [max(1, len(t.split())) for t in texts]
        tokens = [len(scorer.ids(t)) for t in texts]
        ratio = [t / w for t, w in zip(tokens, words)]
        out[name] = {"n": len(texts), "median_words": statistics.median(words),
                     "median_tokens": statistics.median(tokens),
                     "expansion_ratio_mean": round(sum(tokens) / sum(words), 3),
                     "expansion_ratio_median": round(statistics.median(ratio), 3),
                     "expansion_ratio_p95": round(float(np.percentile(ratio, 95)), 3),
                     "truncation_rate_94": round(sum(t > 94 for t in tokens) / len(tokens), 4),
                     "truncation_rate_510": round(sum(t > 510 for t in tokens) / len(tokens), 4)}
    return {"tokenizer": "ytu-ce-cosmos/turkish-medium-bert-uncased (V2 tokenizer)",
            "tokenizer_sha256": sha256(MODEL_DIR / "tokenizer.json"),
            "normalization": "trustlaya.utils.normalize (Turkish I->ı, lowercase)",
            "turkish_source": turkish_note, "groups": out,
            "note": "Token counts exclude [CLS]/[SEP]; V2 reads 94 content tokens."}


def length_analysis(scorer, tt_rows):
    per_row, cue_beyond = [], 0
    for r in tt_rows:
        text = normalize(r["text"])
        encoded = scorer.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        offsets = encoded["offset_mapping"]
        cue = CUE.search(text)
        cue_token = None
        if cue:
            cue_token = next((i for i, (s, e) in enumerate(offsets) if e > cue.start()), None)
        truncated = len(offsets) > 94
        cue_beyond += bool(cue_token is not None and cue_token >= 94)
        per_row.append([r["sample_id"], len(r["text"]), len(r["text"].split()), len(offsets),
                        cue_token, round(cue_token / len(offsets), 3) if cue_token is not None and offsets else None,
                        truncated])
    tokens = [p[3] for p in per_row]
    return {
        "fields": ["sample_id", "chars", "whitespace_tokens", "tokenizer_tokens",
                   "first_cue_token", "first_cue_relative_position", "truncated_at_94"],
        "critical_payload_location_method": "first retrieval-cue regex match (not an annotation of the payload)",
        "summary": {"n": len(per_row), "median_tokens": statistics.median(tokens),
                    "p95_tokens": float(np.percentile(tokens, 95)),
                    "truncated_at_94": sum(p[6] for p in per_row),
                    "truncated_at_510": sum(t > 510 for t in tokens),
                    "with_cue": sum(p[4] is not None for p in per_row),
                    "cue_only_after_token_94": cue_beyond},
        "rows": per_row,
    }


def fit_temperature(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    nll = [float(-np.mean(y * np.log(1 / (1 + np.exp(-logits / t))) +
                          (1 - y) * np.log(1 - 1 / (1 + np.exp(-logits / t)) + 1e-12))) for t in grid]
    return float(grid[int(np.argmin(nll))])


def scale(p, t):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / t))


def main():
    rows = load_rows()
    scorer = Scorer()
    items = attack_items(rows)
    for item in items:
        item["tokens"] = scorer.ids(item["text"])
        item["y"] = int(item["label"] == "ATTACK")
    by = lambda split: [i for i in items if i["split"] == split]
    dev, test, ood = by("DEV"), by("TEST"), by("OOD_TEST")

    # 1. Tokenizer audit.
    (REPORTS / "tokenizer_audit.json").write_text(json.dumps(tokenizer_audit(scorer, rows), indent=2) + "\n")

    # 2. Context ablation (DEV selects, TEST only reported). Same checkpoint throughout.
    grid = [("HEAD", c) for c in CONTEXTS] + [(s, c) for s in ("TAIL", "HEAD_TAIL") for c in CONTEXTS] + \
           [("SLIDING_MAX", 94), ("SLIDING_MAX", 256)]
    ablation = {"threshold": THRESHOLD, "selection_split": "DEV", "grid": [],
                "512": f"not run: BERT has {MAX_POSITIONS} positions including [CLS]/[SEP], so 512 content tokens do not fit; 510 is the native maximum (max_length=512)."}
    scores = {}
    for strategy, capacity in grid:
        entry = {"strategy": strategy, "content_tokens": capacity}
        for name, subset in (("DEV", dev), ("TEST", test), ("OOD_TEST", ood)):
            p, cost = evaluate_strategy(scorer, subset, strategy, capacity)
            scores[(strategy, capacity, name)] = p
            per_source = {}
            for source in sorted({i["source"] for i in subset}):
                idx = [k for k, i in enumerate(subset) if i["source"] == source]
                per_source[source] = metrics([subset[k]["y"] for k in idx], [p[k] for k in idx])
            long_idx = [k for k, i in enumerate(subset) if len(i["tokens"]) > 94]
            entry[name] = {"pooled": metrics([i["y"] for i in subset], p) if name != "OOD_TEST" else None,
                           "per_source": per_source, "cost": cost,
                           "long_inputs_over_94": metrics([subset[k]["y"] for k in long_idx],
                                                          [p[k] for k in long_idx]) if long_idx else None}
        ablation["grid"].append(entry)
        print(f"context {strategy}/{capacity} done", flush=True)
    dev_f1 = {(e["strategy"], e["content_tokens"]): e["DEV"]["pooled"]["f1"] or 0 for e in ablation["grid"]}
    ablation["dev_best_by_f1"] = "/".join(map(str, max(dev_f1, key=dev_f1.get)))
    ablation["note"] = ("V2 was trained at 94 content tokens; longer contexts are a distribution shift for the "
                        "same weights, not a retrained model. dev_best_by_f1 is a DEV observation, not a V5 choice.")
    (REPORTS / "context_ablation.json").write_text(json.dumps(ablation, indent=2) + "\n")
    lines = ["# Context ablation (frozen V2, prompt_injection head, threshold 0.5)", "",
             "Same checkpoint for every row; DEV is the selection split, TEST is reported once. "
             "Tensor Trust rows are attack-only, so pooled FPR comes from Jailbreak/benign sources.", "",
             "| strategy | tokens | DEV F1 | DEV recall | DEV FPR | DEV PR-AUC | DEV ECE | TEST F1 | TEST recall | TEST FPR | TT TEST recall | windows/input (DEV) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in ablation["grid"]:
        d, t = e["DEV"]["pooled"], e["TEST"]["pooled"]
        tt = e["TEST"]["per_source"].get("tensor_trust", {}).get("recall")
        lines.append(f"| {e['strategy']} | {e['content_tokens']} | {d['f1']} | {d['recall']} | {d['fpr']} | "
                     f"{d['pr_auc']} | {d['ece']} | {t['f1']} | {t['recall']} | {t['fpr']} | {tt} | "
                     f"{e['DEV']['cost']['windows_per_input']} |")
    lines += ["", f"512: {ablation['512']}", "", f"DEV best by F1: {ablation['dev_best_by_f1']} (observation only).", ""]
    (REPORTS / "context_ablation.md").write_text("\n".join(lines))

    # 3. V2 public baseline at its native reading (HEAD, 94).
    native = lambda name: scores[("HEAD", 94, name)]
    baseline = {"model": "TrustLaya-S V2 (frozen)", "reading": "HEAD 94 (native)", "threshold": THRESHOLD,
                "score": "raw sigmoid(prompt_injection)", "splits": {}}
    for name, subset in (("DEV", dev), ("TEST", test), ("OOD_TEST", ood)):
        p = native(name)
        baseline["splits"][name] = {source: metrics([subset[k]["y"] for k in idx], [p[k] for k in idx])
                                    for source in sorted({i["source"] for i in subset})
                                    for idx in [[k for k, i in enumerate(subset) if i["source"] == source]]}
    jbb = [r for r in rows if r["source"] == "jailbreakbench" and r["status"] == "included"]
    jbb_p, _ = evaluate_strategy(scorer, [{"tokens": scorer.ids(r["text"])} for r in jbb], "HEAD", 94)
    baseline["jailbreakbench_task_mismatch"] = {
        "note": "Harmful vs benign *requests*, not jailbreak prompts; the injection head is a proxy here.",
        **metrics([int(r["label"] == "HARMFUL_REQUEST") for r in jbb], jbb_p)}
    (REPORTS / "v2_public_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")

    # 4. Calibration audit: temperature fitted on DEV only, applied unchanged to TEST.
    temperature_file = json.loads((MODEL_DIR / "calibration.json").read_text())["prompt_injection"]
    t_dev = fit_temperature([i["y"] for i in dev], native("DEV"))
    calibration = {"head": "prompt_injection", "existing_temperature": temperature_file,
                   "dev_fitted_temperature": round(t_dev, 4), "fit_split": "DEV (pooled, mixed sources)",
                   "threshold_tuning_on_test": False, "per_source": {}}
    for name, subset in (("DEV", dev), ("TEST", test), ("OOD_TEST", ood)):
        p = np.asarray(native(name))
        for source in sorted({i["source"] for i in subset}):
            idx = [k for k, i in enumerate(subset) if i["source"] == source]
            y = [subset[k]["y"] for k in idx]
            calibration["per_source"][f"{name}/{source}"] = {
                "raw": metrics(y, p[idx]), "existing_temperature": metrics(y, scale(p[idx], temperature_file)),
                "dev_temperature": metrics(y, scale(p[idx], t_dev))}
    (REPORTS / "v2_calibration_audit.json").write_text(json.dumps(calibration, indent=2) + "\n")

    # 5. Tensor Trust evaluation and per-attack lengths.
    tt_rows = [r for r in rows if r["source"] == "tensor_trust" and r["status"] == "included"]
    lengths = length_analysis(scorer, tt_rows)
    tt_eval = {"reading": "HEAD 94 (native) plus context grid", "threshold": THRESHOLD,
               "native": {name: baseline["splits"][name].get("tensor_trust") for name in ("DEV", "TEST")},
               "by_native_category_TEST": {}, "context_recall_TEST": {},
               "length_summary": lengths["summary"]}
    p = native("TEST")
    for category in sorted({i["native_category"] for i in test if i["source"] == "tensor_trust"}):
        idx = [k for k, i in enumerate(test) if i["source"] == "tensor_trust" and i["native_category"] == category]
        tt_eval["by_native_category_TEST"][category] = metrics([1] * len(idx), [p[k] for k in idx])
    for e in ablation["grid"]:
        tt_eval["context_recall_TEST"][f"{e['strategy']}/{e['content_tokens']}"] = \
            e["TEST"]["per_source"].get("tensor_trust", {}).get("recall")
    (REPORTS / "tensor_trust_eval.json").write_text(json.dumps(tt_eval, indent=2) + "\n")
    (ROOT / "data/v5_tensor_trust_lengths.json").write_text(json.dumps(
        {k: v for k, v in lengths.items()}, separators=(",", ":")) + "\n")

    # 6. Source generalization: experiment compositions + frozen V2 on each test set.
    lineage = json.loads((ROOT / "data/dataset_lineage.json").read_text())
    score_by_id = {}
    for name, subset in (("DEV", dev), ("TEST", test), ("OOD_TEST", ood)):
        score_by_id.update({i["sample_id"]: s for i, s in zip(subset, native(name))})
    label_by_id = {i["sample_id"]: i["y"] for i in items}
    generalization = {"status": "V5 not trained; only the frozen V2 baseline is measured on each test set.",
                      "experiments": {}}
    for name, spec in lineage["experiments"].items():
        test_ids = [s for s in spec["test"] if s in label_by_id]
        per_source = defaultdict(list)
        for s in test_ids:
            per_source[s.split(":")[0]].append(s)
        generalization["experiments"][name] = {
            "sources": spec["sources"], "note": spec["note"],
            "sizes": {role: len(spec[role]) for role in ("train", "dev", "test")},
            "train_label_counts": dict(Counter(f"{s.split(':')[0]}/{'ATTACK' if label_by_id.get(s) else 'BENIGN'}"
                                               for s in spec["train"] if s in label_by_id)),
            "v2_baseline_test": {src: metrics([label_by_id[s] for s in ids], [score_by_id[s] for s in ids])
                                 for src, ids in per_source.items()},
            "v5_result": None}
    (REPORTS / "source_generalization.json").write_text(json.dumps(generalization, indent=2) + "\n")
    print(f"model windows scored: {scorer.calls}")


if __name__ == "__main__":
    main()
