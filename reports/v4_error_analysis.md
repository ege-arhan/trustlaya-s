# V4 error analysis

Frozen JLL test at selected threshold 0.79: 3370 false positives and 63 false negatives. We stored IDs/hashes and automatic **non-exclusive** heuristic tags for 100 highest-scoring FPs and all 63 FNs; no sensitive raw text was exported. These tags are not human-reviewed intent annotations and do not satisfy a 100-FN human-review requirement because only 63 FNs exist.

| Heuristic tag | First 100 FPs | All 63 FNs |
|---|---:|---:|
| explicit_override | 17 | 1 |
| over_510_tokens | 50 | 8 |
| over_94_tokens | 47 | 30 |
| roleplay | 46 | 12 |
| unclassified_by_heuristic | 3 | 21 |

Spot inspection of ten highest-score FPs found a mix of benign copywriting/persona prompts, explicit generic “ignore previous instructions” wording, and long roleplay. Under a strict untrusted-input policy, some are semantically ambiguous. Ten lowest-score FNs included very short game/place-holder submissions, encoded text, roleplay, and one explicit leetspeak instruction. This is why source labels are not treated as perfect ground truth. The longest JLL bin (>510 tokens) had attack recall 0.976 and FPR 0.860; WINDOW_MAX increases false alarms as the number of windows grows. Human review remains required before a usable intent taxonomy can be claimed.
