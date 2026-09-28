# V3 external error analysis

`scripts/analyze_v3_errors.py` selected up to 100 examples per error type using seed 20260925 and assigned **multi-label regex triage categories**. The full selected IDs and counts are in [v3_error_triage.json](v3_error_triage.json). Categories are heuristic, overlap, and are not independent human annotations. Raw example text remains in gitignored local data. We reviewed representative snippets to verify the dominant mechanisms below.

## TAB DIRECT PERSON/CODE

All **25** v3 window false positives and **92** false negatives were included. Among false positives, 20 contain a titled person (`Mr`, `Mrs`, `Dr`, etc.) and 19 contain legal boilerplate. Many flagged names are public officials or representatives whose TAB label is QUASI/NO_MASK, outside the narrow DIRECT projection. This is a task-definition false alarm, although it could still matter in a broader privacy policy. Among false negatives, 86 contain a titled person and 10 contain a legal application-number pattern. PERSON token F1 is 0.120 and exact PERSON span F1 0.039; the head often recognizes legal numbers but not names in body text. The frozen v2 binary classifier had no span output, so its 0.135 window F1 is not a span comparator.

## JailbreakLLMs

| Error set | Population | Inspected | Selected heuristic counts (overlap allowed) |
|---|---:|---:|---|
| V2 false positives | 4,726 | 100 | 49 long contexts, 48 roleplay phrases, 15 possible regular-label conflicts, 13 override phrases |
| V2 false negatives | 35 | 35 | 20 long contexts, 18 roleplay phrases, 3 code/encoding clues |
| V3 false positives | 267 | 100 | 30 very short, 29 override phrases / possible label conflicts, 27 roleplay phrases, 13 long contexts |
| V3 false negatives | 613 | 100 | 62 long contexts, 59 roleplay phrases, 17 override phrases, 4 code/encoding clues |

Examples of v3 false alarms include one-word messages such as “hello” and “test.” The supposedly regular side also includes at least one explicit “Ignore all I told you” override and an encoded instruction, so some binary labels are noisy. We did **not** flip gold labels after seeing scores. V3 false negatives include long “unrestricted persona,” fictional role, and code-wrapped attack prompts, which are absent from the short Gandalf game-prompt training distribution. The v2 90% false-alarm rate is caused by overgeneralization from synthetic/injection-style training plus a mismatched regular class; this is a supported inference from score distribution and inspected errors, not a proven causal attribution.

At the deployed 94-content-token limit, **594/635 = 93.5%** clean jailbreak positives and **3,873/5,126 = 75.6%** clean regular prompts exceed the input window. Median lengths are 512 and 204 tokenizer tokens respectively. The first-window-only classifier cannot inspect later attack instructions or benign context. This is a concrete mechanism consistent with the long-context error pattern, though it does not by itself explain every false decision.

The controlled benign pool includes only 11 train and 2 dev prompts with explicit attack-discussion terms. This is insufficient to quantify benign-security-discussion behavior. No separately annotated indirect-injection or Turkish error set was available in this run.
