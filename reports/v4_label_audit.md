# V4 label semantics audit

Target operational question: **Does this input attempt to override instructions, extract hidden instructions, or bypass a higher-priority constraint in this context?** Topic words alone are insufficient. We use distinct conceptual labels: DIRECT_ATTACK; BENIGN_SECURITY_DISCUSSION; QUOTED_ATTACK; EDUCATIONAL_CONTENT; SECURITY_DOCUMENTATION; NORMAL_REQUEST. The current external sources do **not** consistently supply these six human-reviewed labels, so the experimental head is trained on a weaker binary projection only.

* JOT JailbreakChat positives are human-authored templates combined with sampled harmful payloads. Duplicate templates were grouped and near-overlap with JLL removed. Their binary attack label is source construction, not per-row human adjudication.
* JOT WildChat negatives are human interactions filtered by the JOT authors. 308 retained negatives contain security-adjacent vocabulary; they are valuable hard negatives, but individual intent has not been manually verified.
* Bordair positives are live-game attempts, yet an isolated short submission can be innocuous outside that game. We used >=500-character rows on DEV and never treated their construction label as a universal direct-jailbreak truth.
* OWASP paragraphs discuss risks and can quote attack wording. They are labeled benign **documentation in full paragraph context**; their domain differs from direct user prompts.
* JailbreakLLMs calls some generic role/persona overrides “regular”; a manually inspected high-scoring regular row began with an explicit “ignore previous instructions” directive for benign copywriting. Several low-scoring positive rows were short, context-dependent game or placeholder text. This is a label/task boundary, not evidence that every flagged row is a true security error.
* deepset's positive label includes broad role changes and format changes. It is evaluated separately as prompt-injection transfer, not pooled with direct jailbreak JLL.

No new six-way human-reviewed annotations were produced. QUOTED_ATTACK and EDUCATIONAL_CONTENT cannot be reliably scored by source-derived binary labels. The inability to distinguish these categories remains a release blocker.
