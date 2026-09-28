# V5 Jev via Vercel: exploratory labeling pilot

Date: 2026-09-25. Status: **SILVER triage only; no GOLD, training, threshold change, or gateway integration.**

## Scope and provenance

- Frozen blind packet: 600 rows, SHA256 `7287e5e7a342b02144270b0757805a8d3aa00204f1d145ddc29e65a5c9f6c862`.
- Sent to Vercel AI Gateway: all **79** rows mapped privately to the OWASP Cheat Sheet Series, and **8** small synthetic controls written for this pilot. No Tensor Trust, Microsoft, or Stack Exchange row was sent.
- OWASP lists the Cheat Sheet Series license as [CC BY-SA 4.0](https://owasp.org/projects/cheat-sheet-series). The blind ID map was used locally for source selection and was never part of API state.
- The 79 OWASP rows carry the corpus label `defensive_documentation_candidate`. This is a provenance label, **not human intent ground truth**. No accuracy, F1, recall, or false-positive rate is claimed for them.
- API: `POST https://ai-gateway.vercel.sh/v1/evaluate`, model alias `typesafe-ai/jev`, based on [Vercel's official evaluation API](https://vercel.com/docs/ai-gateway/modalities/evaluation). The response reported the alias; the underlying version ID was not exposed in the saved responses.
- The API credential was stored briefly in a gitignored, mode-0600 file and removed after the run. It is absent from requests, prediction files, and this report.

## Questions

One `choice` asked for `ATTACK`, `BENIGN_DUAL_USE`, `NORMAL`, or `UNRESOLVED`, explicitly distinguishing operational instructions to the current assistant from discussion/quotation. Three `boolean` questions asked about directed attack, quoted reference, and an **actual** secret value. Questions were identical for every sample. No threshold was selected on this pilot. The exact request schema and per-row responses remain in the ignored private files listed below.

## Measured results

| Cohort | N | API errors | ATTACK | BENIGN_DUAL_USE | NORMAL | UNRESOLVED |
|---|---:|---:|---:|---:|---:|---:|
| OWASP documentation, source label only | 79 | 0 | 8 | 39 | 32 | 0 |
| Synthetic controls, known construction intent | 8 | 0 | 4 | 3 | 1 | 0 |

All **8/8 constructed controls** matched their intended L1 class, including two Turkish controls. This only checks that the request and response mapping work on easy, authored examples; it is not independent performance evidence.

On OWASP, **35/79** rows had a maximum L1 choice probability below 0.80. Among the eight `ATTACK` choices, local inspection found quoted or listed attack phrases inside defensive documentation in all eight. One representative passage began “An attacker could inject …”; its L1 probabilities were ATTACK 0.51 and BENIGN_DUAL_USE 0.48, while the separate `quoted_reference` probability was 0.84. This is the known attack-versus-discussion boundary. The inspection is a single researcher's qualitative review and does not create GOLD labels.

Across the 79 OWASP calls, observed end-to-end client latency was **P50 2,048 ms, P95 2,462 ms, max 3,392 ms**. The calls occurred in two sessions under changing network/provider conditions, so these figures are neither model-only latency nor UNO Q latency. Reported input use was **46,272 tokens**, and the Vercel response reported **$0.00** total gateway cost for these calls on this date. This does not predict future billing.

## Private artifacts

- `benchmarks/v5/private/jev_pilot_request_schema.json`
- `benchmarks/v5/private/jev_pilot_predictions.jsonl` (20 OWASP + 8 controls)
- `benchmarks/v5/private/jev_pilot_summary.json`
- `benchmarks/v5/private/jev_owasp_all_predictions.jsonl` (all 79 OWASP rows)
- Predictions file SHA256: `9253ca916a6de1b29577d0cf0f9e8c358928c10c0e49d44568b41c8b628494f7`

The prediction files contain blind IDs, text hashes, typed answers, usage, and latency. They contain no raw packet text or API credential. The original frozen packet and V2 default are unchanged.

## Decision

Jev successfully returned typed labels through the supplied Vercel credential. The OWASP result shows substantial attack/discussion ambiguity and several likely quoted-example alarms. **Do not promote these labels to GOLD or use them for final benchmark metrics.** A human-reviewed anchor set is required to quantify Jev's error rate and calibrate an abstention threshold. Keep Tensor Trust outside external API runs until its data-use rights are resolved.
