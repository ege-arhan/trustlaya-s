# V5 iteration 5: translation audit (synthetic TRAIN augmentation)

All numbers describe synthetic TRAIN data. None of it is gold, human-labeled or used for testing.
Tokens use the V2 tokenizer. Full numbers: `reports/v5_iter5_short_attack_analysis.json` and below.

| Group | n | median chars | median tokens | tokens/word | question share | "!" share | quote share |
|---|---|---|---|---|---|---|---|
| tr_curated_benign | 600 | 86.5 | 17.0 | 1.369 | 0.042 | 0.005 | 0.097 |
| tr_curated_attack | 150 | 84.5 | 16.0 | 1.459 | 0.007 | 0.0 | 0.033 |
| mt_de_attack | 884 | 127.0 | 52.5 | 2.854 | 0.051 | 0.07 | 0.46 |
| mt_de_benign | 1100 | 47.0 | 18.0 | 2.404 | 0.898 | 0.0 | 0.022 |
| mt_tr_attack | 884 | 105.0 | 25.0 | 2.044 | 0.054 | 0.064 | 0.55 |
| mt_tr_benign | 998 | 41.0 | 9.0 | 1.583 | 0.92 | 0.0 | 0.252 |
| real_de_benign (OASST2/Aya) | 544 | 88.0 | 33.0 | 2.609 | 0.588 | 0.037 | 0.123 |
| real_tr_benign (Aya/OASST2) | 2865 | 36 | 8 | 1.556 | 0.834 | 0.0 | 0.16 |
| english_attack_originals | 884 | 107.5 | 38.0 | 2.453 | 0.072 | 0.074 | 0.543 |

## Same English attack in English / German / Turkish (884 originals)

- Question form kept when the English was a question: German 0.688, Turkish 0.672.
- Median length ratio vs English: characters de 1.131, tr 0.967;
  tokens de 1.286, tr 0.652.
- Braces/placeholders kept: de 0.995, tr 0.949.

## Semantic preservation (back-translation to English, chrF, 300 attacks per language)

| Language | chrF median | chrF 10th percentile | share below 0.40 |
|---|---|---|---|
| de | 0.738 | 0.377 | 0.11 |
| tr | 0.669 | 0.378 | 0.107 |

chrF compares surface characters, so it underestimates meaning preservation for paraphrases and
overestimates it when the model copies English words. It is a screening number, not a quality
judgment. Frequent translated trigrams per source are listed in the JSON sidecar below.

```json
{
 "mt_de_attack": [
  ". . .",
  "nein, nein, nein,",
  "* * *",
  "♪ ♪ ♪",
  "ö ö ö"
 ],
 "mt_de_benign": [
  "was ist der",
  "was sind die",
  "was ist die",
  "was sind einige",
  "was ist ein"
 ],
 "mt_tr_attack": [
  "pls pls pls",
  "lib lib lib",
  "\"ben pwned oldum\"",
  "ben pwned oldum",
  "esnaflib esnaflib esnaflib"
 ],
 "mt_tr_benign": [
  "göz önüne alındığında,",
  "arasındaki fark nedir?",
  "paragraf göz önüne",
  "bu paragraf göz",
  "hangi balık türü?"
 ]
}
```
