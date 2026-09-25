# V5 intent benchmark: acquisition status

**Status: candidate pool, not a validated benchmark.** No V5 model was trained, no v2/v3/v4 default was changed, and there are **zero human-reviewed gold labels** as of 2026-09-25.

The [machine-readable manifest](../benchmarks/v5/dataset_manifest.json) and [row metadata](../benchmarks/v5/dataset_manifest.jsonl) contain provenance and hashes, without raw text. Private raw source snapshots and the 460-row review packet live in `benchmarks/v5/private/` and are gitignored. `scripts/build_v5_review_queue.py` regenerates the queue from pinned/snapshotted sources. It removed 438 normalized exact duplicates from 1,606 acquired rows, leaving **1,168 unique candidates**.

| Source | Unique candidates | Provenance | License / restriction | Proposed role |
|---|---:|---|---|---|
| [Tensor Trust game data](https://github.com/HumanCompatibleAI/tensor-trust-data) revision `747a75e` | 908 | Game attack submissions reported as human; per-row authorship and intent unverified | **Data repository has no explicit license**; local research and review only, no raw redistribution | Unassigned |
| [OWASP Prompt Injection Prevention Cheat Sheet](https://github.com/OWASP/CheatSheetSeries) revision `c04039a` | 79 | Human-authored documentation paragraphs; some contain quoted attack strings | CC BY-SA 4.0 | Unassigned |
| [Microsoft MCP security lesson](https://github.com/microsoft/mcp-for-beginners) revision `5b9b963` | 92 | Human-authored educational paragraphs; per-paragraph authorship unverified | MIT repository | Unassigned |
| [Stack Exchange questions](https://api.stackexchange.com/docs/advanced-search) API 2.3 snapshots | 89 | Public user questions from Stack Overflow / AI Stack Exchange; filtered to posts since 2018-05-02 | CC BY-SA 4.0 with author and post URL retained | Unassigned |

The Stack Exchange [licensing help](https://stackoverflow.com/help/licensing) gives CC BY-SA 4.0 for posts from 2018-05-02 onward. Source files are saved locally and SHA-256 values are in the manifest. Tensor Trust game **source code** is BSD-2-Clause; that does not establish a license for the separate [data repository](https://github.com/HumanCompatibleAI/tensor-trust-data/issues/3). We therefore cannot release its text or train a distributable model on it without resolving rights.

The deterministic review packet contains 200 Tensor Trust candidates and all 260 candidates from the three other source families. It conceals source labels and model scores from annotators. Selection is source and length stratified, not label stratified. Human reviewers must decide whether each text is an actual attack, a quote, a discussion, education, an ordinary request, or ambiguous.

## Source separation and missing pieces

No rows have been assigned to `TRAIN`, `DEV`, `HIDDEN_TEST_A`, or `HIDDEN_TEST_B`. Assigning the current four source families directly would confound source with class: Tensor Trust is attack-enriched while documentation and forum questions are mostly discussion-enriched. A valid source-separated metric requires reviewed attack and benign examples in each evaluated source or a stated cross-source limitation. Candidate [prompt-protection agent-security datasets](https://huggingface.co/datasets/promptprotection/agent-security-datasets) claim human authorship and CC BY 4.0, but their labels are definitional rather than independently reviewed; their contents were not acquired or used here. It may serve as a future held-out source only after provenance and contamination checks.

Normal user requests, Turkish and mixed-language data, matched topic/intent pairs, two distinct hidden sources, and independently reviewed labels remain incomplete. No attack rate or language-specific performance is inferred from the source name.
