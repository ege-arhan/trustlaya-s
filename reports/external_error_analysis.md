# External failure analysis

The first 50 hash-sorted false positives and false negatives per task (where available) were summarized with source labels, score distributions, gold entity types and detector types. No raw legal/person text or community prompt is published. Category assignments below are algorithmic diagnostics, **not independent human adjudication**.

## TAB

| Failure | Available | Inspected | Median score | Source/entity breakdown |
|---|---:|---:|---:|---|
| False positive | 154 | 50 | 0.927 | {} |
| False negative | 198 | 50 | 0.015 | {'PERSON': 53, 'CODE': 31} |

## JailbreakLLMs

| Failure | Available | Inspected | Median score | Source/entity breakdown |
|---|---:|---:|---:|---|
| False positive | 4726 | 50 | 0.948 | {'website': 39, 'discord': 7, 'reddit': 4} |
| False negative | 35 | 35 | 0.180 | {'discord': 21, 'reddit': 10, 'website': 4} |

TAB's dominant measured failure is missed annotated direct PERSON/CODE identifiers (198 false-negative windows); regex evidence has no exact TAB span match. False positives may include QUASI identifiers excluded by this narrow gold mapping. JailbreakLLMs has 4,737 false positives, heavily concentrated in website regular prompts because that source has a much lower attack prevalence; 35 attacks were missed. These are measured errors, not proof of individual intent or downstream attack success.

The benchmark cannot isolate obfuscation, multilingual, indirect injection, or real credential categories because its labels do not annotate these subtypes. No invented subtype counts are reported. Next investigation should obtain human adjudication on a stratified error sample before changing thresholds or training.

## Exploratory human-origin hard negatives

A keyword-selected subset of 842 author-labeled regular community prompts mentions `ignore`, `password`, `token`, `system prompt`, `api key`, `security`, `safety`, or `policy`. 825 were falsely flagged (rate 0.980). This is a post hoc diagnostic slice, not an independent or manually adjudicated benchmark.
