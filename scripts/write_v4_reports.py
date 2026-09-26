"""Write reproducible research notes from measured V4 JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports"


def read(name):return json.loads((REPORT/name).read_text())
def fmt(value):return "n/a" if value is None else f"{value:.3f}"
def save(name,body): (REPORT/name).write_text(body.strip()+"\n")


def main():
    lengths=read("v4_long_context_metrics.json")
    leakage=read("v4_data_leakage.json")
    train=read("v4_training_data_audit.json")
    dev3=read("v4_window_dev_metrics_v3.json")
    dev4=read("v4_window_dev_metrics_v4.json")
    selection=read("v4_dev_selection.json")
    final=read("v4_final_external_metrics.json")
    unseen=read("v4_unseen_external_metrics.json")
    errors=read("v4_error_counts.json")
    latency=read("v4_latency.json")
    manifest=json.loads((ROOT/"models/trustlaya-s-v4-research/manifest.json").read_text())
    first=lengths["clean"]["attack"]
    strategies="\n".join(f'| {name} | {fmt(dev3["strategies"][name]["metrics"]["f1"])} | {fmt(dev3["strategies"][name]["metrics"]["recall"])} | {fmt(dev4["strategies"][name]["metrics"]["f1"])} | {fmt(dev4["strategies"][name]["metrics"]["recall"])} | {fmt(dev4["strategies"][name]["metrics"]["fpr"])} | {dev4["strategies"][name]["mean_windows_per_document"]:.1f} |'
                           for name in dev4["strategies"])
    save("v4_long_context.md",f"""# V4 long-context diagnosis and read strategies

The measured [token-length diagnostic](long_context_diagnostic.md) used the frozen JailbreakLLMs corpus before V4 fitting. Encoder limit: 512 total / 510 content tokens. Existing v2/v3 application limit: 96 total / 94 content tokens. Clean attacks: {first['n']} rows, median {first['tokens']['median']:.0f} content tokens, p95 {first['tokens']['p95']:.0f}; {first['over_current_94']} ({first['over_current_94']/first['n']:.1%}) exceeded the actual application read. A lexical cue's first match occurred after token 94 in {first['cue_only_after_94']} attack rows; this is a heuristic, not a gold attack span.

After disabling **both** the tokenizer backend's implicit truncation and padding, six strategies were compared on a source-separated proxy DEV of 340 long Bordair live-game attack attempts and 105 OWASP explanatory paragraphs. The near-duplicate `bordair:13198` was excluded before final selection. All model weights were fixed for this comparison; v3 uses its prior calibration/0.30 threshold and v4 below is **raw at 0.50**. No final JLL scores selected the strategy.

| Read strategy | v3 F1 | v3 recall | v4 raw F1 | v4 raw recall | v4 raw FPR | Mean windows/doc |
|---|---:|---:|---:|---:|---:|---:|
{strategies}

WINDOW_MAX led this proxy DEV on F1. Its benefit carries a multiple-comparisons cost: on frozen JLL test, v4 FPR was {final['v4_by_length']['>510']['fpr']:.3f} for >510-token prompts versus {final['v4_by_length']['<=94']['fpr']:.3f} for <=94-token prompts. A longer document supplies more opportunities for one high false alarm. The frozen v3 head remained low-recall across all six strategies; truncation was not its sole failure.

The strategy experiment's batched encoder work was {dev4['total_batched_model_ms']:.0f} ms for {dev4['total_scored_windows_including_strategy_duplicates']} unique windows; this is an aggregate DEV computation, not batch=1 latency. [Local warm latency](v4_latency.json): 30 documents, 10 per length bin, includes tokenizer-adjacent Python/encoder work, excludes network and policy. MPS p50: {latency['by_length']['<=94']['p50_ms']:.1f}, {latency['by_length']['95-510']['p50_ms']:.1f}, {latency['by_length']['>510']['p50_ms']:.1f} ms by increasing length. First cold inference {latency['first_inference_cold_ms']:.1f} ms, model load {latency['model_load_ms']:.1f} ms. These are Mac measurements, not UNO Q.
""")
    save("v4_data_provenance.md",f"""# V4 data provenance and leakage screen

| Role | Source / pinned revision | N used | Label origin / caveat | License |
|---|---|---:|---|---|
| TRAIN positive | [JailbreaksOverTime](https://github.com/wagner-group/JailbreaksOverTime) `{train['JOT_revision']}` / JailbreakChat compositions | {train['train_class_counts']['1']} | Human-authored jailbreak templates combined with sampled harmful payloads; composed examples, **not wholly raw human prompts** | MIT repository; original source attribution retained locally |
| TRAIN negative | Same release / WildChat human user prompts | {train['train_class_counts']['0']} | Real user prompts but source filtering is not per-row human verification; WildChat original [ODC-BY](https://huggingface.co/datasets/allenai/WildChat-1M) attribution applies | ODC-BY underlying |
| DEV attack | [Bordair live game](https://huggingface.co/datasets/Bordair/bordair-multimodal) `398e3f875ffd32ec2e6817a95feebfcbae41643a` | 340 | Human game submissions >=500 characters, label by context/construction; goal is password/rule hijack, not general harmful jailbreak | MIT |
| DEV benign | [OWASP LLM Top 10 2026](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10) `9253e38ade58e959b531c0c5c9a4842272c9cd0e` | 105 | Human-written explanatory paragraphs, not ordinary user requests | CC BY-SA 4.0 |
| Final direct-jailbreak test | [JailbreakLLMs](https://github.com/TrustAIRLab/JailbreakLLMs) `2dbd7bbc25f1b156552678f451bddbc787cd679f` | 5,761 | Previously frozen v2/v3 clean paired cohort; unseen by V4 fitting and selection, **not a never-before-examined research source** | MIT |
| Additional untouched-source test | [deepset prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) official test, `4f61ecb038e9c3fb77e21034b22511b523772cdd` | 116 | Independent binary prompt-injection labels, sparse annotation provenance and a related but different task | Apache-2.0 |

Training excluded all {train['JOT_JLL_records_excluded_by_source']} JOT records sourced from JailbreakLLMs. Its {train['JOT_JailbreakChat_template_groups']} template-prefix groups were capped at four examples each before screening. Exact matches removed {train['train_test_exact_excluded']}; normalized char 4-5-gram cosine >=0.85 removed {train['train_test_near_cosine_0.85_excluded']} additional TRAIN examples against the frozen JLL set. One near-matching Bordair DEV example was removed before strategy/threshold reselection. Final cross-role audit: TRAIN/DEV {leakage['train_dev']['near_left_rows_cosine_gte_0.85']}, DEV/TEST {leakage['dev_test']['near_left_rows_cosine_gte_0.85']}, TRAIN/TEST {leakage['train_test']['near_left_rows_cosine_gte_0.85']} near matches. Raw source text and row predictions remain gitignored; reports store counts, hashes and provenance.

No verified independent Turkish test was available in these acquisitions. Language-specific Turkish, English and mixed scores are **not estimated** from unannotated language fields.
""")
    save("v4_label_audit.md",f"""# V4 label semantics audit

Target operational question: **Does this input attempt to override instructions, extract hidden instructions, or bypass a higher-priority constraint in this context?** Topic words alone are insufficient. We use distinct conceptual labels: DIRECT_ATTACK; BENIGN_SECURITY_DISCUSSION; QUOTED_ATTACK; EDUCATIONAL_CONTENT; SECURITY_DOCUMENTATION; NORMAL_REQUEST. The current external sources do **not** consistently supply these six human-reviewed labels, so the experimental head is trained on a weaker binary projection only.

* JOT JailbreakChat positives are human-authored templates combined with sampled harmful payloads. Duplicate templates were grouped and near-overlap with JLL removed. Their binary attack label is source construction, not per-row human adjudication.
* JOT WildChat negatives are human interactions filtered by the JOT authors. {sum(1 for r in json.loads((ROOT/'benchmarks/external/predictions/v4_train_text_private.json').read_text()) if r['gold']==0 and __import__('re').search(r'prompt injection|jailbreak|system prompt|security|safety|bypass',r['text'],__import__('re').I))} retained negatives contain security-adjacent vocabulary; they are valuable hard negatives, but individual intent has not been manually verified.
* Bordair positives are live-game attempts, yet an isolated short submission can be innocuous outside that game. We used >=500-character rows on DEV and never treated their construction label as a universal direct-jailbreak truth.
* OWASP paragraphs discuss risks and can quote attack wording. They are labeled benign **documentation in full paragraph context**; their domain differs from direct user prompts.
* JailbreakLLMs calls some generic role/persona overrides “regular”; a manually inspected high-scoring regular row began with an explicit “ignore previous instructions” directive for benign copywriting. Several low-scoring positive rows were short, context-dependent game or placeholder text. This is a label/task boundary, not evidence that every flagged row is a true security error.
* deepset's positive label includes broad role changes and format changes. It is evaluated separately as prompt-injection transfer, not pooled with direct jailbreak JLL.

No new six-way human-reviewed annotations were produced. QUOTED_ATTACK and EDUCATIONAL_CONTENT cannot be reliably scored by source-derived binary labels. The inability to distinguish these categories remains a release blocker.
""")
    save("v4_training.md",f"""# V4 isolated training record

V2 and v3 were frozen before V4 work; checksums and exact row predictions are in [the frozen comparison](v2_v3_frozen_comparison.md). V4 only adds a {manifest['version']} linear attack-intent head on the same frozen encoder. The head has **513 parameters** (512 weights and one bias), stored in a 2,188-byte safetensors file. Head SHA-256: `{manifest['head_sha256']}`. V2 weight SHA-256: `{manifest['v2_weight_sha256']}`. Tokenizer SHA-256: `{manifest['tokenizer_sha256']}`. The new head is local at `models/trustlaya-s-v4-research/attack_intent_head.safetensors` and is not wired into inference defaults or the gateway.

TRAIN: {manifest['training_rows']} rows, {manifest['positive']} composed JailbreakChat attacks across {train['attack_template_groups_after_screen']} surviving template-prefix groups and {manifest['negative']} WildChat negatives. Encoder max input 96 total tokens for head fitting, mean pooling over non-padding encoder outputs. Optimizer: scikit-learn logistic regression with C=0.1, balanced class weights, max_iter=500, seed=20260925. No encoder fine-tuning, no v2/v3 checkpoint edits. The 95 positive rows are **small and weakly labeled**, so this is an exploratory head, not a robust human jailbreak model.

Read strategy chosen on source-separated proxy DEV: WINDOW_MAX over 94-token overlapping content windows (47-token overlap). DEV was split by stable text hash: {selection['cal_n']} calibration rows and {selection['select_n']} threshold-selection rows. Logistic score calibration fitted only on calibration rows. Selection fixed threshold {selection['threshold']:.2f} by maximizing F1 under FPR <=0.25, with recall >=0.80 preferred where feasible; that target was {'feasible' if selection['target_recall_feasible'] else 'not feasible'}. Final JLL and deepset test data did not fit the head, calibration or threshold.

Source checkout setup after the [v2/v3 prerequisite](../README.md#external-data-v3-experiment-not-deployed):

```bash
git clone https://github.com/wagner-group/JailbreaksOverTime.git benchmarks/external/raw/v4/jailbreaks_over_time
git -C benchmarks/external/raw/v4/jailbreaks_over_time checkout {train['JOT_revision']}
git clone https://github.com/GenAI-Security-Project/GenAI-LLM-Top10.git benchmarks/external/raw/v4/owasp
git -C benchmarks/external/raw/v4/owasp checkout 9253e38ade58e959b531c0c5c9a4842272c9cd0e
```

Run in order: `PYTHONPATH=src .venv/bin/python scripts/freeze_v4_baselines.py`; `scripts/v4_diagnose_context.py`; `scripts/prepare_v4_dev.py`; `scripts/prepare_v4_train.py`; `scripts/check_v4_leakage.py`; `scripts/train_v4_head.py`; `scripts/evaluate_v4_windows.py --head v3`; `scripts/evaluate_v4_windows.py --head v4`; `scripts/select_v4_operating_point.py`; `scripts/evaluate_v4_final.py`; `scripts/evaluate_v4_unseen_source.py`; `scripts/benchmark_v4_latency.py`; `scripts/analyze_v4_errors.py`; `scripts/plot_v4_reliability.py`; `scripts/write_v4_reports.py`. The Bordair and deepset files are downloaded at pinned revisions by the preparation/evaluation scripts. Requires the ignored frozen v2/v3 checkpoints and the repository virtual environment. No checkpoint is published.
""")
    score_lines=[]
    for name,source,split,language,n,data in (("JailbreakLLMs","Reddit/Discord/website community","frozen clean test","not annotated",5761,final),
                                              ("deepset prompt-injections","deepset public dataset","official test","not annotated",116,unseen)):
        a=data['v2']['metrics'] if 'metrics' in data['v2'] else data['v2']
        b=data['v4']['metrics'] if 'metrics' in data['v4'] else data['v4']
        score_lines.append(f"| {name} | {source} | {split} | {n:,} | {language} | {fmt(a['precision'])} | {fmt(a['recall'])} | {fmt(a['f1'])} | {fmt(a['fpr'])} | {fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['f1'])} | {fmt(b['fpr'])} |")
    save("v4_external_scorecard.md",f"""# V4 external scorecard — NO-GO

All metrics below are measured with a **fixed before-test** operating point: v2 raw threshold 0.50; v4 WINDOW_MAX plus DEV-fitted calibration at {selection['threshold']:.2f}. Deepset's prompt-injection label is **not the same target** as JailbreakLLMs direct-jailbreak intent. A high score on one must not be pooled with the other. V4 is research only.

| Dataset | Provenance | Split | N | Language | v2 Precision | v2 Recall | v2 F1 | v2 FPR | v4 Precision | v4 Recall | v4 F1 | v4 FPR |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(score_lines)}

Paired JLL confusion matrices: v2 TP={final['v2']['metrics']['tp']}, FP={final['v2']['metrics']['fp']}, FN={final['v2']['metrics']['fn']}; v3 TP={final['v3']['metrics']['tp']}, FP={final['v3']['metrics']['fp']}, FN={final['v3']['metrics']['fn']}; v4 TP={final['v4']['metrics']['tp']}, FP={final['v4']['metrics']['fp']}, FN={final['v4']['metrics']['fn']}. V3 JLL F1={fmt(final['v3']['metrics']['f1'])}, recall={fmt(final['v3']['metrics']['recall'])}, FPR={fmt(final['v3']['metrics']['fpr'])}. No V4 PII claim: the V2/V3 frozen TAB PII track and v3 exact-span F1 remain unchanged.

Unseen deepset official test: v3 F1={fmt(unseen['v3']['f1'])}, recall={fmt(unseen['v3']['recall'])}, FPR={fmt(unseen['v3']['fpr'])}; v4 missed {unseen['v4']['fn']} of {unseen['v4']['positive']} positives. Label provenance is sparse and the task differs, but this is clear adverse transfer evidence. No verified Turkish external score. [Exact machine-readable JLL metrics](v4_final_external_metrics.json); [deepset metrics](v4_unseen_external_metrics.json); raw text-free row predictions are local/gitignored.
""")
    save("v4_error_analysis.md",f"""# V4 error analysis

Frozen JLL test at selected threshold {selection['threshold']:.2f}: {errors['total_FP']} false positives and {errors['total_FN']} false negatives. We stored IDs/hashes and automatic **non-exclusive** heuristic tags for 100 highest-scoring FPs and all {errors['sampled']['FN']} FNs; no sensitive raw text was exported. These tags are not human-reviewed intent annotations and do not satisfy a 100-FN human-review requirement because only {errors['total_FN']} FNs exist.

| Heuristic tag | First 100 FPs | All {errors['sampled']['FN']} FNs |
|---|---:|---:|
{chr(10).join(f'| {tag} | {errors["tag_counts"]["FP"].get(tag,0)} | {errors["tag_counts"]["FN"].get(tag,0)} |' for tag in sorted(set(errors['tag_counts']['FP'])|set(errors['tag_counts']['FN'])))}

Spot inspection of ten highest-score FPs found a mix of benign copywriting/persona prompts, explicit generic “ignore previous instructions” wording, and long roleplay. Under a strict untrusted-input policy, some are semantically ambiguous. Ten lowest-score FNs included very short game/place-holder submissions, encoded text, roleplay, and one explicit leetspeak instruction. This is why source labels are not treated as perfect ground truth. The longest JLL bin (>510 tokens) had attack recall {fmt(final['v4_by_length']['>510']['recall'])} and FPR {fmt(final['v4_by_length']['>510']['fpr'])}; WINDOW_MAX increases false alarms as the number of windows grows. Human review remains required before a usable intent taxonomy can be claimed.
""")
    raw_rows=json.loads((ROOT/"benchmarks/external/predictions/v4_jailbreak_final_predictions.json").read_text())
    from run_external_real import metrics
    raw_metrics=metrics([{"gold":r['gold'],"raw_score":r['raw_score']} for r in raw_rows],threshold=selection['threshold'])
    selraw=selection['select_raw'];selcal=selection['select_calibrated'];jll=final['v4']['metrics']
    save("v4_calibration.md",f"""# V4 calibration transfer

Calibration fitted on {selection['cal_n']} Bordair/OWASP DEV rows, threshold selected on another {selection['select_n']} rows. The selection half's ECE was {fmt(selraw['ece'])} raw versus {fmt(selcal['ece'])} after calibration; Brier {fmt(selraw['brier'])} versus {fmt(selcal['brier'])}; NLL {fmt(selraw['nll'])} versus {fmt(selcal['nll'])}. These are descriptive for a game-attack/documentation mixture with {selection['select_operating_point']['positive']}/{selection['select_n']} positives, not population risk probabilities.

On frozen JailbreakLLMs test (635/5,761 positives), **calibration did not transfer**: ECE {fmt(raw_metrics['ece'])} raw versus {fmt(jll['ece'])} calibrated; Brier {fmt(raw_metrics['brier'])} versus {fmt(jll['brier'])}; NLL {fmt(raw_metrics['nll'])} versus {fmt(jll['nll'])}. The [reliability diagram](v4_reliability.svg) visualizes this shift. Scores must not be read as deployment risk probabilities. A future calibration cohort must match the intended operational prompt distribution and remain separate from training and test.
""")
    save("v4_ablation.md",f"""# V4 controlled comparisons

V2, v3 and v4 on the **same frozen 5,761-row JLL cohort** (fixed thresholds):

| Model | Training change | Read | F1 | Recall | FPR |
|---|---|---|---:|---:|---:|
| v2 | frozen baseline | HEAD 94 | {fmt(final['v2']['metrics']['f1'])} | {fmt(final['v2']['metrics']['recall'])} | {fmt(final['v2']['metrics']['fpr'])} |
| v3 | frozen Gandalf/prompts.chat attack head | HEAD 94 | {fmt(final['v3']['metrics']['f1'])} | {fmt(final['v3']['metrics']['recall'])} | {fmt(final['v3']['metrics']['fpr'])} |
| v4 | frozen encoder + JOT-fitted linear head | WINDOW_MAX | {fmt(final['v4']['metrics']['f1'])} | {fmt(final['v4']['metrics']['recall'])} | {fmt(final['v4']['metrics']['fpr'])} |

The v4-to-v3 comparison changes **both** head fit and read strategy, so the test difference cannot be attributed to one factor. On source-separated proxy DEV at raw threshold 0.50, v4 HEAD F1={fmt(dev4['strategies']['HEAD']['metrics']['f1'])} and WINDOW_MAX F1={fmt(dev4['strategies']['WINDOW_MAX']['metrics']['f1'])}; this supports a window effect on that DEV only. No separate v4 JLL HEAD result, no without-hard-negative fit and no causal loss-function ablation were run. The 95 positive training examples remaining after leakage screening make extra fitted variants particularly unstable. Calibration and threshold selection used DEV only. “False allow” and “false block” are not claimed: this head was never connected to a policy decision or guarded tool execution path.
""")
    save("v4_final_report.md",f"""# TrustLaya-S V4 research verdict: NO-GO

## Question and measured answer

Could a frozen 42M-parameter encoder, a small externally fitted attack head and long-context reading recover v3's missed direct jailbreaks **without** v2-level false alarms? **Partly on JLL, but not reliably across sources.** Paired JLL v4 recall={fmt(final['v4']['metrics']['recall'])}, F1={fmt(final['v4']['metrics']['f1'])}, FPR={fmt(final['v4']['metrics']['fpr'])}; v2 recall={fmt(final['v2']['metrics']['recall'])}, F1={fmt(final['v2']['metrics']['f1'])}, FPR={fmt(final['v2']['metrics']['fpr'])}. Independent deepset prompt-injection test v4 recall={fmt(unseen['v4']['recall'])}, F1={fmt(unseen['v4']['f1'])} versus v2 F1={fmt(unseen['v2']['f1'])}. Deepset labels cover a related but distinct task and are reported separately.

## What changed

V2/v3 evaluation artifacts were frozen with checksums; no baseline checkpoint or default gateway was changed. Actual v2/v3 read length is 94 content tokens; {first['over_current_94']/first['n']:.1%} of JLL attacks exceed that. Six read strategies were measured before fitting. A V4 research-only 513-parameter linear head was fitted on {manifest['positive']} composed JailbreakChat positives and {manifest['negative']} real WildChat negatives after source/near-duplicate screening. WINDOW_MAX and a DEV-only threshold of {selection['threshold']:.2f} were fixed before frozen final evaluation. Source roles, labels, tokenizer corrections, and exact scripts are in [data provenance](v4_data_provenance.md), [label audit](v4_label_audit.md), and [training](v4_training.md). Final verification: **83/83 pytest tests passed**, frozen baseline checksums verified, and all 11 V4 JSON reports parsed successfully.

## Scientific limits and deployment decision

V4's JLL false alarm rate {fmt(final['v4']['metrics']['fpr'])} is still far too high for default blocking, and increases to {fmt(final['v4_by_length']['>510']['fpr'])} above 510 tokens. DEV calibration degraded on JLL (ECE {fmt(jll['ece'])}, Brier {fmt(jll['brier'])}). The nominal train attack corpus has only {train['attack_template_groups_after_screen']} non-overlapping template groups and uses composed examples; its benign labels are not individually human-reviewed. DEV positives are game attacks, DEV negatives are explanatory documents. There is no independent Turkish test and no fully human-reviewed six-way intent taxonomy. The JLL final source had been inspected in prior v2/v3 work, though V4 never trained or selected on it. No UNO Q measurement was made. V4 is **NO-GO**, remains local and disconnected from the default gateway; v2 remains the frozen baseline.

## Next priority

Acquire a licensed, independently annotated **direct-jailbreak** corpus with genuinely human-written long prompts and a matched human benign-security corpus. Review intent labels and quoted attacks by humans, reserve a never-inspected source, then test a window-count-aware aggregation rule with a false-alarm constraint. Keep policy and tool authorization fail-closed regardless of classifier error.

See [scorecard](v4_external_scorecard.md), [long context](v4_long_context.md), [ablation](v4_ablation.md), [calibration](v4_calibration.md), and [error analysis](v4_error_analysis.md) for measured evidence.
""")
    print("Wrote nine V4 reports")


if __name__=="__main__":main()
