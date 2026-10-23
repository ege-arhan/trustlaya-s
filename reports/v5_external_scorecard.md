# V5 external scorecard

| Dataset | Provenance | Split | N | Language | v2 Precision / Recall / F1 / FPR | v4 Precision / Recall / F1 / FPR | V5 |
|---|---|---|---:|---|---|---|---|
| JailbreakLLMs clean paired | Public Reddit/Discord/website corpus; source labels, previously examined | Historical frozen comparison | 5,761 | Not independently reviewed by language | 0.115 / 0.946 / 0.206 / 0.898 | 0.145 / 0.901 / 0.250 / 0.657 | No model |
| deepset prompt-injections official test | Public prompt-injection task, distinct from direct jailbreak | Historical independent source for V4 | 116 (60 positive) | Not independently reviewed by language | See [V4 scorecard](v4_external_scorecard.md) | 4/60 detected; F1 0.125 | No model |
| V5 candidate pool | Human game submissions, documentation, forum questions; **not row reviewed** | Unassigned | 1,168 unique | 908 undetermined game rows; documents mostly English | Not scored | Not scored | Gold labels 0 |
| V5 hidden A/B | Not yet acquired/frozen | Hidden | 0 | Unknown | Not scored | Not scored | Not scored |

The old V4 proxy DEV eight-strategy comparison is [separate](v5_context_strategies.md). Its source-derived labels and 76.4% positive composition make it unsuitable as a headline external score. There is no validated Turkish score, attack/discussion confusion rate, or V5 generalization result. No benchmark here changes the default gateway.
