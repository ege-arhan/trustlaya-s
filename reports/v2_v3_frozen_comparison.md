# Frozen v2/v3 baseline for v4

This snapshot was made from commit `508483f` before v4 model or data experiments. The exact row predictions, metrics, calibration and threshold files are copied into the gitignored local `benchmarks/external/predictions/v4_frozen_inputs/`. Those copies are read-only. The tracked [manifest](v4_frozen_manifest.json) and table below provide SHA-256 verification. Model and tokenizer files are referenced by checksum; neither is overwritten.

## Fixed paired external comparisons

| Dataset / granular task | Clean N | Version | Precision | Recall | F1 | FPR | Threshold / calibration |
|---|---:|---|---:|---:|---:|---:|---|
| TAB DIRECT PERSON/CODE window presence | 1,794 | v2 frozen | 0.115 | 0.165 | 0.135 | 0.077 | 0.50 raw |
| TAB DIRECT PERSON/CODE window presence | 1,794 | v3 NO-GO | 0.306 | 0.107 | 0.158 | 0.015 | 0.47 Platt-calibrated; token threshold 0.50 raw |
| JailbreakLLMs community direct-jailbreak proxy | 5,761 | v2 frozen | 0.115 | 0.946 | 0.206 | 0.898 | 0.50 raw |
| JailbreakLLMs community direct-jailbreak proxy | 5,761 | v3 NO-GO | 0.076 | 0.035 | 0.048 | 0.052 | 0.30 Platt-calibrated |

The full original v2 cohorts remain 2,079 TAB and 5,888 JailbreakLLMs rows. The smaller paired cohorts exclude the exact/near overlaps listed in `v3_exclusions.json`. V2/v3 exact predictions are not regenerated from a changed pipeline for this comparison. TAB is window presence here; v3 exact-span F1 was 0.079 and must not be mixed with this table.

## Fixed source versions and models

- TAB Git revision `558e09e26d6b36f5f78440074e6a233946d98bd9` (official ECHR test).
- JailbreakLLMs Git revision `2dbd7bbc25f1b156552678f451bddbc787cd679f` (Reddit/Discord/website subset).
- v2 model SHA-256 `99a8527de00fed3a520d136d26cdda9acc79dff2fae5c725ef773159b565563c`.
- v3 PII head SHA-256 `cea63619596a031c3c7e4f868d92c8ef0170b6a4a180ec6c05de832ed0a45cf3`; attack head SHA-256 `b207389c55cc79c1807d5b4ef160be7bb8dc44eebc9b180f946427e08bc0e174`; the v2 encoder is shared and frozen.
- v2 tokenizer SHA-256 `d1982e608b0883c5840b3dc9a735af3fbc45f3b75417dea44fa3efab3854346f`; v3 tokenizer SHA-256 `d1982e608b0883c5840b3dc9a735af3fbc45f3b75417dea44fa3efab3854346f`.
- Detailed configuration, calibration and threshold JSONs are in the snapshot below.

| Snapshot file | SHA-256 |
|---|---|
| `encoder_config.json` | `ec59f7b60b5cd3e4e842dd0a685929c3d4a6a0498d27987824dd885f0090b0ad` |
| `v2_calibration.json` | `d96930cda27a12d3f3436706ef53015351a595b0aef5c152a69375077ce06f96` |
| `v2_jailbreak_predictions.json` | `bd2454d22ecba54fef0e245ef207a7d65f9029f73fecbe9af6fc236356a5224b` |
| `v2_metrics.json` | `300708a6bf3f74969b845b561f3ef663777831cb05aa4fdc59a60e136d97f7b9` |
| `v2_policy.yaml` | `6475167cded698ed4671ee4fdcb5714dce27a0e7f4b351876fd88ed521c68e30` |
| `v2_tab_predictions.json` | `06230d6aa26c5130045b21f640ee8dd4f85e6b40b07ca0515468810b1e7c4ab2` |
| `v2_thresholds.json` | `92abf11e9dca2f84c6d12ff99f07c355c1e556bd94e7b0c8dd6d91683957532c` |
| `v3_calibration.json` | `c7613186ac6ce8764f38b094d18c9cb11abe70f54cd0ac5394cf82cb2459b53d` |
| `v3_exclusions.json` | `a5e7100d67797a9c8b00f8129973652ad30e94cf56c00bfe8463c56f1dd3f1d1` |
| `v3_jailbreak_predictions.json` | `951b3b1c0ccc4690310681946fa51e9e329ab8a91e577496444089bc9c3481bc` |
| `v3_metrics.json` | `61dffb66d04ca7366a8e536e59106465fbf150df2a42a8f5b3853faa1fc3604e` |
| `v3_model_manifest.json` | `9645b386c6c1a328e74c8ae9f7730e785650b740f4f5a5194ef9d598b9b6dcee` |
| `v3_tab_predictions.json` | `7c02a7b1ee877880b6c66e3bcc8238f46b788e24526d9f3db2a55d2339718677` |
| `v3_thresholds.json` | `7f239b192aa3fb089e9e6388d0c6e7d0704a19ff51c232b523dc0da791fda849` |

`python scripts/freeze_v4_baselines.py` verifies this snapshot on future runs and refuses mismatched copies. External-test reuse after v3 iteration means the old JailbreakLLMs set is a frozen comparison, **not a new blind v4 final test**. V4 requires a separately sourced unseen final set.
