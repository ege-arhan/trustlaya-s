"""V3 external evaluation; thresholds/calibration are selected on DEV only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

from trustlaya.model import TrustLaya, BACKBONE
from trustlaya.v3_external import binary_metrics, calibration_metrics, decode_spans, canonicalize_spans, fingerprint, span_metrics
from run_external_real import LOCAL, tab_samples, jailbreak_samples
from train_v3_external import OUT, ROOT, benign_prompts, gandalf, encode, pii_arrays


def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(OUT)
    model = TrustLaya(BACKBONE, pretrained=False)
    weights = ROOT / "models/trustlaya-s-v2/model.safetensors"
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == manifest["v2_weight_sha256"]
    for filename, field in (("pii_token_head.safetensors", "pii_head_sha256"),
                            ("attack_head.safetensors", "attack_head_sha256")):
        assert hashlib.sha256((OUT / filename).read_bytes()).hexdigest() == manifest[field]
    model.load(weights); model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    token = torch.nn.Linear(model.encoder.config.hidden_size, 5)
    token.load_state_dict(load_file(OUT / "pii_token_head.safetensors")); token.eval()
    attack = torch.nn.Linear(model.encoder.config.hidden_size, 1)
    attack.load_state_dict(load_file(OUT / "attack_head.safetensors")); attack.eval()
    return model, tokenizer, token, attack, device


def cached_features(name, model, tokenizer, texts, device, token_features):
    path = LOCAL / "v3_cache" / f"{name}.npy"
    if path.exists(): return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = encode(model, tokenizer, texts, device, token_features=token_features)
    np.save(path, values)
    return values


def pii_score(features, head):
    logits = []
    with torch.inference_mode():
        for batch in torch.from_numpy(features.astype(np.float32)).split(128):
            logits.append(torch.softmax(head(batch), -1).numpy())
    return np.concatenate(logits)


def attack_score(features, head):
    with torch.inference_mode():
        return torch.sigmoid(head(torch.from_numpy(features.astype(np.float32)))).flatten().numpy()


def fit_calibration(raw, gold):
    raw = np.clip(np.asarray(raw), 1e-5, 1 - 1e-5)
    logit = np.log(raw / (1 - raw)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, random_state=20260925, max_iter=1000)
    model.fit(logit, gold)
    return {"slope": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])}


def apply_calibration(raw, fit):
    raw = np.clip(np.asarray(raw), 1e-5, 1 - 1e-5)
    z = fit["slope"] * np.log(raw / (1 - raw)) + fit["intercept"]
    return 1 / (1 + np.exp(-z))


def select_threshold(gold, score, max_fpr):
    curve = []
    for t in np.arange(.01, 1, .01):
        m = binary_metrics(gold, score, float(t))
        curve.append({"threshold": round(float(t), 2), **m})
    feasible = [m for m in curve if m["fpr"] is not None and m["fpr"] <= max_fpr]
    best = max(feasible, key=lambda m: (m["f1"] or 0, m["recall"] or 0, -m["threshold"]))
    return best["threshold"], curve


def token_predictions(prob, offsets, valid, threshold, texts=None):
    out = []
    for p, off, mask in zip(prob, offsets, valid, strict=True):
        labs = p.argmax(-1).tolist()
        for j in range(len(labs)):
            if not mask[j] or max(p[j, 1:]) < threshold: labs[j] = 0
        spans = decode_spans(labs, off)
        out.append(canonicalize_spans(texts[len(out)], spans) if texts is not None else spans)
    return out


def token_metrics(gold_labels, prob, valid, threshold):
    pred = prob.argmax(-1)
    pred[(prob[:, :, 1:].max(-1) < threshold) | ~valid] = 0
    g = gold_labels[valid]; p = pred[valid]
    result = {}
    for name, ids in (("ALL", (1, 2, 3, 4)), ("PERSON", (1, 2)), ("CODE", (3, 4))):
        gg = np.isin(g, ids); pp = np.isin(p, ids)
        tp = int((gg & pp).sum()); fp = int((~gg & pp).sum()); fn = int((gg & ~pp).sum())
        result[name] = {"tp": tp, "fp": fp, "fn": fn,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    return result


def doc_metrics(rows, scores, threshold):
    docs = {}
    for r, score in zip(rows, scores, strict=True):
        doc = docs.setdefault(r["doc_id"], {"gold": 0, "score": 0.})
        doc["gold"] = max(doc["gold"], r["gold"])
        doc["score"] = max(doc["score"], float(score))
    return binary_metrics(np.array([x["gold"] for x in docs.values()]),
                          np.array([x["score"] for x in docs.values()]), threshold)


def inspect_errors(rows, score, threshold, task, limit=100):
    selected = []
    for r, s in zip(rows, score, strict=True):
        wrong = "FP" if r["gold"] == 0 and s >= threshold else "FN" if r["gold"] == 1 and s < threshold else None
        if wrong:
            selected.append({"sample_id": r["sample_id"], "text_hash": fingerprint(r["text"]),
                             "source": r["source"], "kind": wrong, "score": round(float(s), 6),
                             "text": r["text"], "task": task})
    # Worst confidently wrong examples first, then deterministic sampling for coverage.
    selected.sort(key=lambda r: (-r["score"] if r["kind"] == "FP" else r["score"], r["sample_id"]))
    return {kind: [r for r in selected if r["kind"] == kind][:limit] for kind in ("FP", "FN")}


def main():
    model, tokenizer, pii_head, attack_head, device = load_model()
    leak = json.loads((ROOT / "reports/v3_leakage_results.json").read_text())
    def overlap_ids(key): return {r["sample_id"] for r in leak[key]["overlaps"]}
    dev, _ = tab_samples(tokenizer, "dev")
    dev = [r for r in dev if r["sample_id"] not in overlap_ids("TAB_train_dev")]
    # Case-disjoint calibration and operating-point halves.
    pcal = [r for r in dev if int(hashlib.sha256(r["doc_id"].encode()).hexdigest()[:8], 16) % 2 == 0]
    pselect = [r for r in dev if int(hashlib.sha256(r["doc_id"].encode()).hexdigest()[:8], 16) % 2 != 0]
    pdata = {}
    for split, rows in (("cal", pcal), ("select", pselect)):
        features = cached_features(f"tab_dev_{split}_tokens", model, tokenizer, [r["text"] for r in rows], device, True)
        prob = pii_score(features, pii_head)
        labels, valid, offsets = pii_arrays(rows, tokenizer)
        score = np.where(valid, prob[:, :, 1:].max(-1), 0).max(-1)
        pdata[split] = (rows, prob, score, labels, valid, offsets)
    pfit = fit_calibration(pdata["cal"][2], np.array([r["gold"] for r in pcal]))
    psel_score = apply_calibration(pdata["select"][2], pfit)
    pthreshold, pcurve = select_threshold(np.array([r["gold"] for r in pselect]), psel_score, .10)
    token_choices = []
    for value in np.arange(.05, .96, .05):
        token_choices.append((float(value), token_metrics(pdata["select"][3], pdata["select"][1], pdata["select"][4], float(value))["ALL"]["f1"] or 0))
    ptoken_threshold = max(token_choices, key=lambda pair: (pair[1], pair[0]))[0]
    print("PII DEV window threshold", pthreshold, "token threshold", ptoken_threshold, "calibration", pfit, flush=True)

    benign, _ = benign_prompts()
    positive = gandalf("validation")
    # Independent prompt split for calibration and threshold selection.
    acal_text = []; acal_y = []; asel_text = []; asel_y = []
    for i, (text, y) in enumerate([(t, 1) for t in positive] + [(t, 0) for t in benign["validation"]]):
        if f"dev:{i}" in overlap_ids("prompt_train_dev"): continue
        bucket = int(hashlib.sha256(fingerprint(text).encode()).hexdigest()[:8], 16) % 2
        (acal_text if bucket == 0 else asel_text).append(text)
        (acal_y if bucket == 0 else asel_y).append(y)
    acal_raw = attack_score(cached_features("attack_dev_cal_pooled", model, tokenizer, acal_text, device, False), attack_head)
    asel_raw = attack_score(cached_features("attack_dev_select_pooled", model, tokenizer, asel_text, device, False), attack_head)
    afit = fit_calibration(acal_raw, np.array(acal_y))
    asel_score = apply_calibration(asel_raw, afit)
    athreshold, acurve = select_threshold(np.array(asel_y), asel_score, .25)
    print("Attack DEV threshold", athreshold, "calibration", afit, flush=True)

    # Frozen IDs, gold and predictions define the test cohort. No training or selection uses these rows.
    frozen_tab = json.loads((LOCAL / "v2_frozen_tab_predictions.json").read_text())
    frozen_jail = json.loads((LOCAL / "v2_frozen_jailbreak_predictions.json").read_text())
    tab_ids = {r["sample_id"] for r in frozen_tab}; jail_ids = {r["sample_id"] for r in frozen_jail}
    test_tab = [r for r in tab_samples(tokenizer, "test")[0] if r["sample_id"] in tab_ids]
    test_jail = [r for r in jailbreak_samples()[0] if r["sample_id"] in jail_ids]
    assert len(test_tab) == len(frozen_tab) == 2079 and len(test_jail) == len(frozen_jail) == 5888
    assert all(r["gold"] == x["gold"] for r, x in zip(test_tab, frozen_tab, strict=True))
    assert all(r["gold"] == x["gold"] for r, x in zip(test_jail, frozen_jail, strict=True))
    blocked_tab = overlap_ids("TAB_train_test") | overlap_ids("TAB_dev_test")
    blocked_jail = overlap_ids("prompt_train_jailbreak_test") | overlap_ids("prompt_dev_jailbreak_test")
    test_tab = [r for r in test_tab if r["sample_id"] not in blocked_tab]
    test_jail = [r for r in test_jail if r["sample_id"] not in blocked_jail]
    frozen_tab = [r for r in frozen_tab if r["sample_id"] not in blocked_tab]
    frozen_jail = [r for r in frozen_jail if r["sample_id"] not in blocked_jail]
    assert len(test_tab) == len(frozen_tab) and len(test_jail) == len(frozen_jail)
    ptest_feature = cached_features("tab_test_tokens", model, tokenizer, [r["text"] for r in test_tab], device, True)
    ptest_prob = pii_score(ptest_feature, pii_head)
    ptest_labels, ptest_valid, ptest_offsets = pii_arrays(test_tab, tokenizer)
    ptest_raw = np.where(ptest_valid, ptest_prob[:, :, 1:].max(-1), 0).max(-1)
    ptest_score = apply_calibration(ptest_raw, pfit)
    pspans = token_predictions(ptest_prob, ptest_offsets, ptest_valid, ptoken_threshold,
                               [r["text"] for r in test_tab])
    gold_spans = [list(dict.fromkeys(map(tuple, r["gold_spans"]))) for r in test_tab]
    atest_feature = cached_features("jail_test_pooled", model, tokenizer, [r["text"] for r in test_jail], device, False)
    atest_raw = attack_score(atest_feature, attack_head)
    atest_score = apply_calibration(atest_raw, afit)
    gandalf_test = [t for i, t in enumerate(gandalf("test"))
                    if f"gandalf:{i}" not in (overlap_ids("prompt_train_gandalf_test") | overlap_ids("prompt_dev_gandalf_test"))]
    gtest_feature = cached_features("gandalf_test_pooled", model, tokenizer, gandalf_test, device, False)
    gtest_score = apply_calibration(attack_score(gtest_feature, attack_head), afit)

    # Save exact row predictions without raw content. Error-analysis text is local only.
    LOCAL.mkdir(parents=True, exist_ok=True)
    for name, rows, raw, cal, threshold in (("tab", test_tab, ptest_raw, ptest_score, pthreshold),
                                            ("jailbreak", test_jail, atest_raw, atest_score, athreshold)):
        payload = [{"sample_id": r["sample_id"], "text_hash": fingerprint(r["text"]), "gold": r["gold"],
                    "source": r["source"], "raw_score": float(a), "calibrated_score": float(b),
                    "predicted_label": int(b >= threshold)}
                   for r, a, b in zip(rows, raw, cal, strict=True)]
        (LOCAL / f"v3_{name}_predictions.json").write_text(json.dumps(payload, indent=2) + "\n")
    errors = {"PII": inspect_errors(test_tab, ptest_score, pthreshold, "PII"),
              "JAILBREAK": inspect_errors(test_jail, atest_score, athreshold, "JAILBREAK")}
    (LOCAL / "v3_error_samples_private.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2) + "\n")
    result = {"model": "TrustLaya-S v3 experimental", "thresholds_from_dev": {"pii": pthreshold, "pii_token": ptoken_threshold, "attack": athreshold},
              "calibration_from_dev": {"pii": pfit, "attack": afit},
              "dev": {"PII_cal_n": len(pcal), "PII_select_n": len(pselect),
                      "attack_cal_n": len(acal_y), "attack_select_n": len(asel_y),
                      "PII_select": binary_metrics(np.array([r["gold"] for r in pselect]), psel_score, pthreshold),
                      "attack_select": binary_metrics(np.array(asel_y), asel_score, athreshold),
                      "PII_calibration_raw": calibration_metrics(np.array([r["gold"] for r in pcal]), pdata["cal"][2]),
                      "PII_calibration_fit": calibration_metrics(np.array([r["gold"] for r in pcal]), apply_calibration(pdata["cal"][2], pfit)),
                      "attack_calibration_raw": calibration_metrics(np.array(acal_y), acal_raw),
                      "attack_calibration_fit": calibration_metrics(np.array(acal_y), apply_calibration(acal_raw, afit))},
              "test": {"PII_window": binary_metrics(np.array([r["gold"] for r in test_tab]), ptest_score, pthreshold),
                       "PII_token": token_metrics(ptest_labels, ptest_prob, ptest_valid, ptoken_threshold),
                       "PII_exact_span": span_metrics(gold_spans, pspans),
                       "PII_document": doc_metrics(test_tab, ptest_score, pthreshold),
                       "PII_calibration_raw": calibration_metrics(np.array([r["gold"] for r in test_tab]), ptest_raw),
                       "PII_calibration_fit": calibration_metrics(np.array([r["gold"] for r in test_tab]), ptest_score),
                       "JAILBREAK": binary_metrics(np.array([r["gold"] for r in test_jail]), atest_score, athreshold),
                       "GANDALF_attack_only_detection": {"n": len(gandalf_test), "detected": int((gtest_score >= athreshold).sum()),
                                                          "fraction": float((gtest_score >= athreshold).mean())},
                       "JAILBREAK_calibration_raw": calibration_metrics(np.array([r["gold"] for r in test_jail]), atest_raw),
                       "JAILBREAK_calibration_fit": calibration_metrics(np.array([r["gold"] for r in test_jail]), atest_score)},
              "paired_v2_on_clean_subset": {
                  "PII_window": binary_metrics(np.array([r["gold"] for r in frozen_tab]), np.array([r["raw_score"] for r in frozen_tab]), .5),
                  "JAILBREAK": binary_metrics(np.array([r["gold"] for r in frozen_jail]), np.array([r["raw_score"] for r in frozen_jail]), .5)},
              "leakage_excluded": {"TAB_test": len(blocked_tab), "JAILBREAK_test": len(blocked_jail)},
              "error_sample_counts": {task: {k: len(v) for k, v in typ.items()} for task, typ in errors.items()}}
    reports = ROOT / "reports"; reports.mkdir(exist_ok=True)
    (reports / "v3_external_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (reports / "v3_threshold_curves.json").write_text(json.dumps({"PII_DEV": pcurve, "ATTACK_DEV": acurve}, indent=2, allow_nan=False) + "\n")
    (OUT / "calibration.json").write_text(json.dumps(result["calibration_from_dev"], indent=2) + "\n")
    (OUT / "decision_thresholds.json").write_text(json.dumps(result["thresholds_from_dev"], indent=2) + "\n")
    print(json.dumps(result["test"], indent=2), flush=True)


if __name__ == "__main__": main()
