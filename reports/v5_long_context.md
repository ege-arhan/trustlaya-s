# V5 candidate length diagnostic

Tokenizer SHA-256 is recorded in `benchmarks/v5/dataset_manifest.json`. The v2/v3/v4 encoder has 512 total positions, leaving 510 content tokens with two special tokens. The current single-pass inference uses 96 total / **94 content tokens**. These counts use the actual tokenizer and normalization used by TrustLaya-S. Candidate text remains private.

| Original content tokens | Unique candidates |
|---|---:|
| 0–128 | 687 |
| 129–256 | 169 |
| 257–384 | 94 |
| 385–512 | 79 |
| 513–768 | 113 |
| 769–1024 | 14 |
| 1025–1536 | 5 |
| 1537+ | 7 |

Of **1,168** unique candidates, **612 (52.4%)** exceed the current 94-content-token reading limit. Median length is 100, p90 535.5, p95 621.3, p99 1019.2 content tokens. The metadata records original length, model-visible length, truncation, and bucket per row. The attack and benign rates **per bucket are unavailable** because no row is human reviewed. Coarse attack location is likewise unavailable; keyword positions would not be gold attack spans.

The separate, already frozen historical JLL diagnostic found 94.5% of attacks over 94 content tokens; see [v4 long context](v4_long_context.md). That statistic comes from old source labels and does not establish the causal share of V5 errors. The new reviewed benchmark will compare errors for attacks at beginning/middle/end with 94-token and windowed reading.
