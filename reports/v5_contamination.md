# V5 candidate contamination audit

`python scripts/audit_v5_candidates.py` checked the 1,168 unique candidate texts against the pinned v2 training split (6,972 rows), v4 train (2,173), v4 development (445), and historical clean JLL (5,761). It used normalized exact hashes and hashed character 4–5-gram cosine similarity with a 0.85 flag threshold. The [machine results](../benchmarks/v5/contamination.json) include local sample IDs and no text.

| Prior corpus | Normalized exact overlap | V5 candidate rows with near similarity ≥0.85 |
|---|---:|---:|
| v2 train | 0 | 0 |
| v4 train | 0 | 0 |
| v4 DEV | 0 | **1** |
| historical JLL clean | 0 | 0 |

The builder discarded **438 normalized exact duplicate source rows** before the audit. The single near-overlapping v4 DEV candidate is excluded from the 600-row human review packet and must be excluded from any future headline metric. No candidate cross-source near pair was flagged at this threshold. This is a lexical similarity screen, not a proof of no semantic or embedding-level contamination. No `TRAIN`/`DEV`/hidden roles or hidden source snapshots exist yet; cross-split, MinHash/LSH and embedding checks are pending source assignment. A threshold-selected hidden test must be re-audited without reading examples manually.
