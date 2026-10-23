# V5 iteration 4: dataset research (Turkish, German, multilingual, short-question attacks)

Searched Hugging Face (queries: turkish/german/multilingual prompt injection, jailbreak, PolyGuard,
turkish conversation), GitHub repositories, and known multilingual safety datasets. Every
candidate's card was read; provenance was checked, not inferred from the name. Machine-readable
list with revisions: [`data/v5_iter4_manifest.json`](../data/v5_iter4_manifest.json).

**Result: no real, human-written Turkish or German prompt-injection/jailbreak source exists among
the public candidates found.** Every Turkish set is author-constructed (synthetic, template
expansion or agent-generated), machine-translated, or derived from deepset. German attack text
exists only as machine translations or undocumented multilingual sets. Inside the human sources
already used, native non-English attacks are negligible (Tensor Trust 16 de / 1 tr, JailbreakLLMs
3 / 3, HackAPrompt 8 tr).

| Dataset | Revision | Languages | License | Size | Provenance (from card/files) | Verdict |
|---|---|---|---|---|---|---|
| [3nesdeniz/turkish-conversation-prompt-injection](https://huggingface.co/datasets/3nesdeniz/turkish-conversation-prompt-injection) | `29d7593984` | tr | CC BY 4.0 (card + LICENSE) | 750 | synthetic, author-written (card: 'The text is synthetic and was produced for this dataset') | no: synthetic (allowed only as a later, flagged TRAIN-only stage) |
| [3nesdeniz/turkish-prompt-injection-1k](https://huggingface.co/datasets/3nesdeniz/turkish-prompt-injection-1k) | `1cbd1152d9` | tr | CC BY 4.0 | 1000 | synthetic: human-designed templates expanded deterministically | no: template-generated |
| [Jensy1234/turkish-conversation-prompt-injection](https://huggingface.co/datasets/Jensy1234/turkish-conversation-prompt-injection) | `792984cecf` | tr | CC BY 4.0 | 750 | mirror of 3nesdeniz/turkish-conversation-prompt-injection | no: duplicate of the above |
| [AltaySec/turkish-llm-injection](https://huggingface.co/datasets/AltaySec/turkish-llm-injection) | `713a82c38d` | tr | CC BY 4.0 | 300 | 120 hand-written + 180 agent-generated payloads (card) | no: partly LLM-generated, attack-only |
| [fevziegeyurtsevenler/turkish-prompt-injection](https://huggingface.co/datasets/fevziegeyurtsevenler/turkish-prompt-injection) | `fae488a164` | tr | CC BY 4.0 | 107 | author technique list with defenses | no: tiny technique catalogue, superseded by PolyGuardBench |
| [fevziegeyurtsevenler/multilingual-prompt-injection](https://huggingface.co/datasets/fevziegeyurtsevenler/multilingual-prompt-injection) | `9cfe5ab6c6` | en, tr | CC BY 4.0 | 217 | author technique list with {SECRET}/{URL} placeholders | no: templates with placeholders |
| [fevziegeyurtsevenler/multilingual-jailbreak](https://huggingface.co/datasets/fevziegeyurtsevenler/multilingual-jailbreak) | `e8b4dbdd2e` | en, tr | CC BY 4.0 | 27 | author technique list | no: 27 rows |
| [fevziegeyurtsevenler/PolyGuardBench](https://huggingface.co/datasets/fevziegeyurtsevenler/PolyGuardBench) | `45886b8e3e` | tr, en | CC BY 4.0 | n/a | consolidates the author's guardrail-arena and over-refusal sets | no: same author-constructed material as above, not collected in the wild |
| [OnerAYTAS/Turkish_prompt_injection_jailbreak_dataset](https://huggingface.co/datasets/OnerAYTAS/Turkish_prompt_injection_jailbreak_dataset) | `444e9b6ae4` | tr (+en) | CC BY-NC-SA 4.0 (non-commercial) | several CSVs | machine translation (translategemma:4b, Google Translate) of English sets | no: machine-translated and non-commercial |
| [beratcmn/turkish-prompt-injections](https://huggingface.co/datasets/beratcmn/turkish-prompt-injections) | `c40c38f8ca` | tr | Apache-2.0 | 662 | translation of deepset/prompt-injections (card) | NO: derived from the deepset OOD test (leakage) |
| [hoatac/prompt-injections-turkish](https://huggingface.co/datasets/hoatac/prompt-injections-turkish) | `f24a40827a` | tr | none stated | n/a | no card | no: no license or provenance |
| [tljohnsilver/zn-prompt-injection-bench](https://huggingface.co/datasets/tljohnsilver/zn-prompt-injection-bench) | `6ae877560d` | en + 8 MT locales incl. de | CC BY 4.0 | 23699 | English seeds + machine-translated attacks; includes 394 deepset rows | no: German side is machine-translated and it contains deepset |
| [yanismiraoui/prompt_injections](https://huggingface.co/datasets/yanismiraoui/prompt_injections) | `bd55359f2f` | en, fr, de, es, it, pt, ro | Apache-2.0 | ~1,000 | annotations_creators: no-annotation; generation not documented | no: undocumented provenance, attack-only |
| [rikka-snow/prompt-injection-multilingual](https://huggingface.co/datasets/rikka-snow/prompt-injection-multilingual) | `f1ad1f3dd4` | vi, en, zh, fr, de, hi | MIT | n/a | no card text | no: undocumented provenance |
| [Octavio-Santana/prompt-injection-attack-detection-multilingual](https://huggingface.co/datasets/Octavio-Santana/prompt-injection-attack-detection-multilingual) | `2a5a015d04` | multi | GPL | n/a | merge of Injection-Attack-Detection-Dataset + rikka-snow | no: derived merge, copyleft, undocumented base provenance |
| [Necent/llm-jailbreak-prompt-injection-dataset](https://huggingface.co/datasets/Necent/llm-jailbreak-prompt-injection-dataset) | `4edfb5aeaa` | multi | MIT (gated) | large | union of 30+ public sources (incl. ones already used) | no: re-packaging; overlap and deepset leakage risk |
| [darkknight25/Multilingual_Jailbreak_Dataset](https://huggingface.co/datasets/darkknight25/Multilingual_Jailbreak_Dataset) | `0f32d0818c` | en, hi, ru, fr, zh, de, es | MIT | 700 | scenario prompts, generation not documented | no: undocumented, no benign side |
| [ToxicityPrompts/PolyGuardPrompts](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardPrompts) | `c5b466a95b` | 17 languages incl. de (no tr) | CC BY 4.0 | 29K | natural + machine-translated safety prompts | no: different task (harmful content) |
| [DAMO-NLP-SG/MultiJail](https://huggingface.co/datasets/DAMO-NLP-SG/MultiJail) | `-` | en, zh, it, vi, ar, ko, th, bn, sw, jv | MIT | 315 x 10 | human translations of harmful questions | no: no de/tr, different task |
| [MollyShuu/MLJailDe (GitHub)](https://github.com/MollyShuu/MLJailDe) | `-` | multi | none | n/a | generated from a paper's pipeline | no: generated |
| [Lakera/mosscap_prompt_injection](https://huggingface.co/datasets/Lakera/mosscap_prompt_injection) | `-` | mostly en | MIT | 279K | human game submissions | no: no reliable attack label (checked in iteration 3) |

## Gates

- **GATE 1 (real Turkish/German attack source): FAIL.** None found. The best Turkish candidate,
  `3nesdeniz/turkish-conversation-prompt-injection`, is well documented and pairs every attack with
  a benign boundary row, but its card states the text is synthetic. Under the iteration-4 rules it
  may only enter a later, explicitly flagged TRAIN-only stage, never a final TEST.
- **GATE 2 (short question-form attack source): PARTIAL.** HackAPrompt (already used) is human and
  short, but rarely question-shaped; no independent human question-form injection set was found.

Consequence: experiments R5–R8 (new Turkish/German and question-form attacks) cannot be built from
real public data. They were not trained. See the master report for what was run instead.
