# V5 iteration 4: attack sources

| Source | Status in V5 | Language | Human-written | Label rule |
|---|---|---|---|---|
| Tensor Trust (raw dump v2 + v1 benchmarks) | TRAIN/DEV/TEST since V5 v1 | mostly English (16 de, 1 tr detected) | yes (game) | upstream success heuristics / benchmarks |
| JailbreakLLMs 2023-12-25 | TRAIN/DEV/TEST since V5 v1 | mostly English | yes (community) | author jailbreak flag |
| HackAPrompt | TRAIN/DEV/TEST since iteration 3 | English (8 tr detected) | yes (competition) | `correct == True` only |
| Gandalf ignore_instructions | OOD test only | English | yes (game) | source: all attacks |
| deepset prompt-injections | OOD test only, never trained on | en + de (~1/3 de) | curated, provenance per row undocumented | source label |
| New Turkish / German human attack source | **none found** | – | – | – |

No new attack source was added in iteration 4. Candidates and reasons for rejection:
[v5_iter4_dataset_research.md](v5_iter4_dataset_research.md).
