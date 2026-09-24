"""Generate public aggregate reports from private local prediction archives."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_external_real import ROOT, LOCAL, REPORT, jailbreak_samples, metrics

R=json.loads((REPORT/"external_results.json").read_text())
B=json.loads((REPORT/"external_baseline_results.json").read_text()) if (REPORT/"external_baseline_results.json").exists() else {}
T=json.loads((LOCAL/"tab_predictions.json").read_text())
J=json.loads((LOCAL/"jailbreak_predictions.json").read_text())

def f(x):return "—" if x is None else f"{x:.3f}"

def table_line(name,m):
    return f"| {name} | {m['n']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['fpr'])} | {f(m['fnr'])} | {f(m['roc_auc'])} | {f(m['pr_auc'])} |"

def bootstrap_f1(rows, cluster=False, reps=1000):
    rng=np.random.default_rng(20260925)
    groups={}
    for r in rows:groups.setdefault(r["doc_id"] if cluster else r["sample_id"],[]).append(r)
    keys=list(groups)
    vals=[]
    for _ in range(reps):
        sample=[r for k in rng.choice(keys,size=len(keys),replace=True) for r in groups[k]]
        tp=sum(r["gold"]==1 and r["predicted_label"]==1 for r in sample)
        fp=sum(r["gold"]==0 and r["predicted_label"]==1 for r in sample)
        fn=sum(r["gold"]==1 and r["predicted_label"]==0 for r in sample)
        vals.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return [float(v) for v in np.percentile(vals,[2.5,97.5])]

def main():
    tab=R["TAB"]["raw"]; jail=R["JailbreakLLMs"]["raw"]
    tab_ci=bootstrap_f1(T,True); jail_ci=bootstrap_f1(J)
    lines=["# TRUSTLAYA-S EXTERNAL REAL-WORLD EVALUATION","",
           "| Dataset | Provenance | N | Language | Task | TrustLaya-S Precision | TrustLaya-S Recall | TrustLaya-S F1 | FPR | FNR | ROC-AUC / PR-AUC |",
           "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|",
           f"| TAB official test | Real public ECHR legal cases with anonymization gold | {tab['n']} windows / {R['TAB']['source_stats']['documents']} cases | English | Narrow DIRECT PERSON/CODE window PII | {f(tab['precision'])} | {f(tab['recall'])} | {f(tab['f1'])} | {f(tab['fpr'])} | {f(tab['fnr'])} | {f(tab['roc_auc'])} / {f(tab['pr_auc'])} |",
           f"| JailbreakLLMs community subset | Human/community collected prompts, author labeled | {jail['n']} prompts | Not annotated per row | Jailbreak proxy via prompt-injection head | {f(jail['precision'])} | {f(jail['recall'])} | {f(jail['f1'])} | {f(jail['fpr'])} | {f(jail['fnr'])} | {f(jail['roc_auc'])} / {f(jail['pr_auc'])} |",
           "","**Interpretation:** These are separate tasks and must not be averaged. The main evaluation uses external datasets absent from the recorded TrustLaya-S training text after the documented exact/near-duplicate screen. Internal synthetic project tests are regression/stress tests only. No clinical PHI or real-secret F1 is claimed.",
           "","## Frozen setup","",
           f"- v2 weight SHA-256: `{R['frozen_weight_sha256']}`; ONNX CPU; deployed tokenizer and 96-token model input; fixed raw-score threshold `{R['fixed_threshold']}`. Model weights and thresholds were not changed.",
           "- Long TAB cases were split into nonoverlapping 94-wordpiece windows with two special tokens. Gold span offsets were checked against source text. A boundary-crossing identifier drops the window. Only one quality-checked annotator was used per case. This window-level task is different from document-level PII detection.",
           "- TAB positive means a fully contained gold `DIRECT` `PERSON` or `CODE` mention. Other `QUASI` identifiers are outside this narrow target; predicted detections on them count as false positives in this metric. The classification head is broader than this target, so this is a domain/task-transfer diagnostic, not an exhaustive PII capability estimate.",
           "- JailbreakLLMs excludes `open_source` repository prompts; `reddit`, `discord`, and `website` are retained. A jailbreak request and a malicious instruction embedded in tool output are different tasks. The head was evaluated as a proxy, not advertised as a purpose-built jailbreak classifier.",
           "- Both model heads truncate each evaluated window/prompt at 96 model tokens. Especially for long jailbreak prompts, this can hide decisive later content. The jailbreak BERT baseline reads up to 512 tokens, so the model comparison also reflects different input budgets.",
           "- The deployed Turkish-first normalization is applied unchanged to English text, including mapping capital `I` to dotless `ı`. This is part of the frozen pipeline and may contribute to English domain shift.",
           f"- TAB source commit `558e09e26d6b36f5f78440074e6a233946d98bd9`; JailbreakLLMs source commit `2dbd7bbc25f1b156552678f451bddbc787cd679f`. Source repositories and licenses are in the [dataset audit](external_benchmark_dataset_audit.md).",
           "","## Sampling uncertainty","",
           f"TAB F1 95% cluster-bootstrap interval across 127 source cases: {tab_ci[0]:.3f}–{tab_ci[1]:.3f}. Jailbreak F1 95% row-bootstrap interval: {jail_ci[0]:.3f}–{jail_ci[1]:.3f}. Each uses 1,000 seeded resamples; the jailbreak interval does not account for source-family clustering and may be optimistic.",
           "","## Contamination and exclusions","",
           "Training comparison used `data/splits/train.jsonl` and the pinned Turkish PII training source (12,082 total texts). Normalized exact matching and char 4–5-gram cosine similarity ≥0.85 were screened. No training exact/near matches were found among evaluated rows. This is a text-overlap screen; it cannot exclude all semantic contamination or unrecorded encoder pretraining.",
           "","| Dataset | Candidates | Conflicting-label rows removed | Within-benchmark duplicates removed | Training exact removed | Training near removed | Retained |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for n in ("TAB","JailbreakLLMs"):
        c=R[n]["contamination"]
        lines.append(f"| {n} | {c['candidate']} | {c.get('label_conflict_removed',0)} | {c.get('within_benchmark_duplicate',0)} | {c.get('training_exact',0)} | {c.get('training_near_0.85',0)} | {c['retained']} |")
    lines += ["","TAB also dropped 4 boundary-crossing windows before duplicate screening. JailbreakLLMs excluded 213 `open_source` rows before duplicate screening. Identical text with conflicting gold labels was entirely removed; keeping one label arbitrarily would bias the result.",
              "","## Source breakdown: community jailbreak prompts","",
              "| Source | N | Positive | Precision | Recall | F1 | FPR | FNR |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in R["JailbreakLLMs"]["by_source"].items():
        lines.append(f"| {name} | {m['n']} | {m['positive']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['fpr'])} | {f(m['fnr'])} |")
    lines += ["","## Span evidence, separate from the classifier","",
              f"The existing pattern-assisted extractor exactly matched {R['TAB']['span_exact']['tp']} of {R['TAB']['span_exact']['tp']+R['TAB']['span_exact']['fn']} retained gold direct PERSON/CODE spans (exact-span recall {f(R['TAB']['span_exact']['recall'])}). This is not a token-level NER model. A policy decision cannot be interpreted as successful entity extraction.",
              "","## Fixed-threshold sweep (diagnostic; no test-set selection)","",
              "| Dataset | Threshold | Precision | Recall | F1 | FPR | FNR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for n in ("TAB","JailbreakLLMs"):
        sweep=R[n]["threshold_sweep"]
        for target in (.10,.25,.50,.75,.90):
            key=min(sweep,key=lambda k:abs(float(k)-target));m=sweep[key]
            lines.append(f"| {n} | {float(key):.2f} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['fpr'])} | {f(m['fnr'])} |")
    lines += ["","Full 0.05–0.95 sweep and confusion counts are in `external_results.json`. The 0.50 operating point was fixed before evaluation; no sweep point has been promoted to deployment.",
              "","## Access and task gaps","",
              "- i2b2/n2c2 clinical PHI and SecretBench require authorized agreements; access was unavailable. MIMIC-IV-Note also lacks PHI gold for this task. No clinical PHI or real-secret score is reported.",
              "- Open de-identified glaucoma notes have no PHI gold, so a true clinical FPR/F1 cannot be inferred. Authoritative data-governance documents have no two-person human gold annotation here; governance remains unevaluated.",
              "- TAB is English. JailbreakLLMs has no per-row language gold, so language-specific F1 cannot be computed reliably; the external headline makes no Turkish generalization claim. Prior Turkish project scores remain internal results.",
              "- AgentDojo/BIPIA/InjecAgent are constructed agentic benchmarks and remain a separate tier, not mixed with this human-origin text headline.",
              "","## Reproduction","",
              "From this repository with its base `.venv` installed, a single command clones the pinned public sources, installs baseline-only dependencies, checks model/data versions and regenerates all results:",
              "","```bash","bash scripts/run_external_all.sh","```","","Individual stages, if needed:","","```bash","PYTHONPATH=src .venv/bin/python scripts/run_external_real.py",
              "PYTHONPATH=src .venv/bin/python scripts/run_external_baselines.py",
              "PYTHONPATH=src .venv/bin/python scripts/run_neuraltrust_baseline.py",
              "PYTHONPATH=src .venv/bin/python scripts/audit_jailbreak_baseline.py",
              "PYTHONPATH=src .venv/bin/python scripts/calibrate_external_tab.py",
              "PYTHONPATH=src .venv/bin/python scripts/run_external_ablation.py",
              "PYTHONPATH=src .venv/bin/python scripts/run_clinical_unlabeled.py",
              "PYTHONPATH=src .venv/bin/python scripts/write_external_report.py","```","",
              "Prediction archives under `benchmarks/external/predictions/` contain hashes, IDs, scores and evidence offsets, never raw text, and are intentionally gitignored. The source corpora are also gitignored. Official sources, versions, local dependencies and caveats are recorded in the dataset audit."]
    ablation_path=REPORT/"external_ablation.json"
    if ablation_path.exists():
        a=json.loads(ablation_path.read_text())
        lines += ["","## Frozen component diagnostics","",
                  "| Dataset | Component | F1 | FPR | FNR |","|---|---|---:|---:|---:|"]
        for name,parts in a.items():
            for key in ("model_raw","model_existing_calibration","model_calibration_evidence","policy_non_allow_operational_proxy"):
                m=parts[key];lines.append(f"| {name} | {key} | {f(m['f1'])} | {f(m['fpr'])} | {f(m['fnr'])} |")
        lines.append("")
        for name,parts in a.items():
            lines.append(f"{name}: {parts['abstain_count']} abstentions; policy actions {parts['policy_action_counts']}.")
        lines += ["","`policy_non_allow_operational_proxy` counts REDACT, REVIEW and BLOCK as interventions across all risk tasks; its F1 is **not** a PII or jailbreak classifier score. Evidence boosts only the existing PII patterns. No component was fitted on the test set."]
    clinical_path=REPORT/"external_clinical_unlabeled.json"
    if clinical_path.exists():
        c=json.loads(clinical_path.read_text())
        lines += ["","## De-identified clinical note diagnostic (unlabeled)","",
                  f"The public glaucoma repository contains {c['documents']} de-identified notes. Across {c['windows']} nonoverlapping model windows, raw PII score ≥0.50 occurred in {c['pii_raw_flagged_windows']} windows and {c['pii_raw_flagged_documents']} documents. **These are flag counts, not false-positive rates.** The corpus has medication annotations but no PHI gold; surrogate or residual identifiers may remain. No clinical PHI precision, recall or F1 is inferred."]
    (REPORT/"external_real_world_benchmark.md").write_text("\n".join(lines)+"\n")

    base=["# Compatible external baselines","","Same retained sample IDs, same gold, fixed 0.50 score threshold. A baseline score is a model or recognizer score, not a calibrated risk probability.","",
          "| Task | System | N | Precision | Recall | F1 | FPR | FNR |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in (("TAB narrow PII",tab),("Jailbreak proxy",jail)):
        base.append(f"| {name} | TrustLaya-S v2 frozen | {m['n']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['fpr'])} | {f(m['fnr'])} |")
    for key,task in (("presidio","TAB narrow PII"),("jailbreak_bert","Jailbreak classification")):
        if key in B:
            v=next(v for v in B[key].values() if isinstance(v,dict));base.append(f"| {task} | {B[key]['model']} | {v['n']} | {f(v['precision'])} | {f(v['recall'])} | {f(v['f1'])} | {f(v['fpr'])} | {f(v['fnr'])} |")
    neural_path=REPORT/"external_neuraltrust_baseline.json"
    if neural_path.exists():
        neural=json.loads(neural_path.read_text());m=neural["metrics"]
        base.append(f"| Jailbreak classification | {neural['model']} | {m['n']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['fpr'])} | {f(m['fnr'])} |")
    base += ["","Presidio uses default English recognizers and spaCy `en_core_web_lg` 3.8.0; relevant PERSON/identifier entity types were requested. Exact TAB `DIRECT PERSON/CODE` is a narrower target than Presidio's detection taxonomy. No baseline threshold was tuned on this test set.",
             "","Meta Prompt Guard 86M is manual-gated and was not accessible. ProtectAI DeBERTa v2's model card says it does not detect jailbreak attacks, so comparing it on the jailbreak benchmark would conflate tasks. Secret scanners were not run because authorized real-secret gold was unavailable. No baseline values were imputed."]
    overlap_path=REPORT/"external_baseline_overlap.json"
    if overlap_path.exists():
        o=json.loads(overlap_path.read_text());c=o["overlap"]
        base += ["","## BERT training-source contamination","",
                 f"The BERT model card identifies `{o['baseline_training_dataset']}` as training data, whose card cites the same jailbreak prompt repository as this evaluation. We screened both published train CSVs at revision `{o['revision']}` ({o['training_rows_loaded']} rows). Of {c['candidate']} evaluated prompts, {c.get('training_exact',0)} exact and {c.get('training_near_0.85',0)} near training matches were removed for the shared clean-row comparison. Even the remaining rows share a source family and are **not a source-independent test of BERT**.","",
                 "| System | Clean N | Precision | Recall | F1 | FPR | FNR |","|---|---:|---:|---:|---:|---:|---:|"]
        for system,m in o["same_clean_rows"].items():
            base.append(f"| {system} | {m['n']} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {f(m['fpr'])} | {f(m['fnr'])} |")
        base += ["","Do not use the unfiltered BERT row above for a model ranking. The clean-row comparison is a contamination diagnostic and still has source-family leakage risk."]
    if neural_path.exists():
        base += ["","## NeuralTrust provenance limit","",
                 f"`{neural['model']}` is MIT licensed and explicitly classifies direct jailbreaks; revision `{neural['revision']}` was run on the same retained prompts. Its model card says it was fine-tuned on a **private** dataset, so training overlap cannot be verified. Its score is a compatible-task reference with **unknown contamination status**, not an independent clean baseline or leaderboard claim. It reads up to 512 tokens while TrustLaya-S reads 96."]
    (REPORT/"external_baselines.md").write_text("\n".join(base)+"\n")

    cal=["# External calibration diagnostic","","Raw probabilities and the existing internally fitted calibration were evaluated on the external test sets. **No temperature was fitted on either external test set.** These probabilities are task-model scores, not verified real-world risk likelihoods.","",
         "| Dataset | Score | ECE (10 bins) | Brier | NLL |","|---|---|---:|---:|---:|"]
    for n in ("TAB","JailbreakLLMs"):
        for label,key in (("Raw","raw"),("Existing internal temperature","calibrated_existing")):
            m=R[n][key];cal.append(f"| {n} | {label} | {f(m['ece'])} | {f(m['brier'])} | {f(m['nll'])} |")
    cal += ["","The existing temperature does not change 0.50 classifications here. On this domain-shifted jailbreak set it worsens all three calibration metrics. The 10-bin reliability data below allow a diagram to be reproduced without plotting raw text.","",
            "| Dataset | Bin | N | Mean raw score | Observed positive rate |","|---|---:|---:|---:|---:|"]
    for n,rows in (("TAB",T),("JailbreakLLMs",J)):
        for b in range(10):
            sub=[r for r in rows if min(int(r["raw_score"]*10),9)==b]
            if sub:cal.append(f"| {n} | {b/10:.1f}–{(b+1)/10:.1f} | {len(sub)} | {np.mean([r['raw_score'] for r in sub]):.3f} | {np.mean([r['gold'] for r in sub]):.3f} |")
    dev_path=REPORT/"external_tab_dev_calibration.json"
    if dev_path.exists():
        d=json.loads(dev_path.read_text())
        cal += ["","## Post-baseline TAB dev experiment","",
                f"A temperature of {d['temperature']:.3f} was fitted on {d['dev_n_after_test_overlap']} official TAB dev windows after training-data contamination and exact dev/test-overlap filtering. Near dev/test overlap was not screened. This did not modify deployed model files.","",
                "| Split | Score | ECE | Brier | NLL |","|---|---|---:|---:|---:|"]
        for split,label,key in (("Dev","Raw","dev_raw"),("Dev","Dev-fitted","dev_fitted"),("Test","Raw","test_raw"),("Test","Dev-fitted","test_dev_fitted")):
            m=d[key];cal.append(f"| {split} | {label} | {f(m['ece'])} | {f(m['brier'])} | {f(m['nll'])} |")
    cal += ["","JailbreakLLMs has no official held-out calibration split, so no new external temperature was fitted there. TAB test labels were never used to fit the dev-only calibrator."]
    (REPORT/"external_calibration.md").write_text("\n".join(cal)+"\n")

    err=["# External failure analysis","","The first 50 hash-sorted false positives and false negatives per task (where available) were summarized with source labels, score distributions, gold entity types and detector types. No raw legal/person text or community prompt is published. Category assignments below are algorithmic diagnostics, **not independent human adjudication**.",""]
    for n,rows in (("TAB",T),("JailbreakLLMs",J)):
        err += [f"## {n}","", "| Failure | Available | Inspected | Median score | Source/entity breakdown |","|---|---:|---:|---:|---|"]
        for name,cond in (("False positive",lambda r:r["gold"]==0 and r["predicted_label"]==1),
                          ("False negative",lambda r:r["gold"]==1 and r["predicted_label"]==0)):
            sub=sorted([r for r in rows if cond(r)],key=lambda r:r["text_hash"])
            inspect=sub[:50]
            if n=="TAB" and name=="False negative":
                breakdown=Counter(typ for r in inspect for _,_,typ in r.get("gold_spans",[]))
            elif n=="TAB":
                breakdown=Counter(e["type"] for r in inspect for e in r["evidence"])
            else:breakdown=Counter(r["source"] for r in inspect)
            err.append(f"| {name} | {len(sub)} | {len(inspect)} | {f(float(np.median([r['raw_score'] for r in inspect])) if inspect else None)} | {dict(breakdown)} |")
        err += [""]
    err += ["TAB's dominant measured failure is missed annotated direct PERSON/CODE identifiers (198 false-negative windows); regex evidence has no exact TAB span match. False positives may include QUASI identifiers excluded by this narrow gold mapping. JailbreakLLMs has 4,737 false positives, heavily concentrated in website regular prompts because that source has a much lower attack prevalence; 35 attacks were missed. These are measured errors, not proof of individual intent or downstream attack success.",
            "","The benchmark cannot isolate obfuscation, multilingual, indirect injection, or real credential categories because its labels do not annotate these subtypes. No invented subtype counts are reported. Next investigation should obtain human adjudication on a stratified error sample before changing thresholds or training."]
    # Exploratory hard negatives are an exact subset of the author-labeled real
    # regular prompts. Keyword selection is disclosed and has no new gold labels.
    import re
    prompt_lookup={s["sample_id"]:s["text"] for s in jailbreak_samples()[0]}
    hard=[r for r in J if r["gold"]==0 and re.search(r"\b(?:ignore|password|token|system prompt|api key|security|safety|policy)\b",prompt_lookup[r["sample_id"]],re.I)]
    if hard:
        flagged=sum(r["predicted_label"] for r in hard)
        err += ["","## Exploratory human-origin hard negatives","",
                f"A keyword-selected subset of {len(hard)} author-labeled regular community prompts mentions `ignore`, `password`, `token`, `system prompt`, `api key`, `security`, `safety`, or `policy`. {flagged} were falsely flagged (rate {flagged/len(hard):.3f}). This is a post hoc diagnostic slice, not an independent or manually adjudicated benchmark."]
    (REPORT/"external_error_analysis.md").write_text("\n".join(err)+"\n")

    with (REPORT/"dataset_provenance.csv").open("w",newline="") as fcsv:
        fields=["dataset","source_url","paper","license","access_type","real_or_synthetic","human_generated","deidentified","label_source","sample_count","used_in_training","contamination_rate","notes"]
        w=csv.DictWriter(fcsv,fieldnames=fields,lineterminator="\n");w.writeheader()
        w.writerow(dict(dataset="TAB official test",source_url="https://github.com/NorskRegnesentral/text-anonymization-benchmark",paper="https://doi.org/10.1162/coli_a_00458",license="MIT",access_type="public",real_or_synthetic="real ECHR legal cases",human_generated="yes",deidentified="public legal text; identifiers can remain",label_source="manual annotators, quality-checked",sample_count=tab["n"],used_in_training="no recorded text overlap",contamination_rate="0 training exact/near; 9 within-test duplicates removed",notes="127 cases; narrow DIRECT PERSON/CODE windows"))
        w.writerow(dict(dataset="JailbreakLLMs community",source_url="https://github.com/TrustAIRLab/JailbreakLLMs",paper="https://arxiv.org/abs/2308.03825",license="MIT",access_type="public",real_or_synthetic="community-collected human prompts",human_generated="yes",deidentified="not applicable",label_source="researcher-authored jailbreak/regular labels",sample_count=jail["n"],used_in_training="no recorded text overlap",contamination_rate=f"0 training exact/near; {R['JailbreakLLMs']['contamination'].get('label_conflict_removed',0)} conflicting rows and {R['JailbreakLLMs']['contamination'].get('within_benchmark_duplicate',0)} duplicates removed",notes="213 prompt-repository rows excluded; per-row language not annotated"))
        clinical_path=REPORT/"external_clinical_unlabeled.json"
        if clinical_path.exists():
            c=json.loads(clinical_path.read_text())
            w.writerow(dict(dataset="Glaucoma clinical notes",source_url="https://github.com/jche253/Glaucoma_Med_Dataset",paper="see source README",license="BSD-3-Clause",access_type="public",real_or_synthetic="real clinical notes",human_generated="yes",deidentified="yes; PHI replaced by asterisks",label_source="medication labels only; no PHI gold",sample_count=c["documents"],used_in_training="not known",contamination_rate="not measured for unlabeled diagnostic",notes="PII flag counts only; never headline F1/FPR"))
        for name,url,note in (("i2b2 2014","https://n2c2.dbmi.hms.harvard.edu/data-sets","DUA; not evaluated"),("SecretBench","https://github.com/setu1421/SecretBench","DPA; not evaluated")):
            w.writerow(dict(dataset=name,source_url=url,paper="see dataset audit",license="restricted",access_type="agreement required",real_or_synthetic="real deidentified" if name.startswith("i2b2") else "real public-code candidates",human_generated="yes",deidentified="yes" if name.startswith("i2b2") else "no",label_source="human gold",sample_count="not evaluated",used_in_training="unknown",contamination_rate="not measured",notes=note))

if __name__=="__main__":main()
