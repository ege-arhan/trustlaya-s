# TrustLaya-S

## V5 Jev-only SILVER annotation

The current V5 experiment uses Jev via Vercel AI Gateway to label a **separate 600-row SILVER corpus** with zero human labels. Run `.venv/bin/python scripts/run_v5_jev_silver.py` after setting `AI_GATEWAY_API_KEY`, then `.venv/bin/python scripts/summarize_v5_jev_silver.py`. The resumable predictions and raw source texts stay under gitignored `benchmarks/v5/private/`; the public manifest contains only aggregates and hashes. This is **not a human GOLD benchmark or a measured model accuracy result**. Of the original frozen 600-row packet, 260 license-documented rows are included. Its 340 Tensor Trust rows are excluded from the external API run because the data repository has no explicit license; 340 separate rows from the pinned deepset *train* split take their place. See the [Jev-only report](reports/v5_jev_only_silver.md) and the [frozen V2 diagnostic on these labels](reports/v5_jev_v2_diagnostic.md). To rerun the latter locally, use `.venv/bin/python scripts/evaluate_v5_jev_silver.py`; it does not train or change the default gateway.

## V5 kör insan anotasyonu (yerel altyapı)

Önceden oluşturulan yerel insan anotasyon arayüzü ayrı bir deney olarak korunmuştur; mevcut Jev-only iş akışı onu kullanmaz. Ayrıntılar için [Türkçe kullanım kılavuzu](docs/v5_annotation_ui_tr.md). Yeni model eğitilmedi. V2 varsayılan olarak kalır.

## V4 long-context research (NO-GO; local only)

The [V4 final report](reports/v4_final_report.md) records a source-separated experiment on the frozen encoder. On the same 5,761-row JailbreakLLMs external cohort, V4 reached F1 0.250 and recall 0.901, but its false-alarm rate was 0.657. On the independent, related deepset prompt-injection test, V4 recall was 0.067. DEV calibration failed to transfer to JailbreakLLMs. **V4 is not published and is not connected to the default CLI, API, policy engine or gateway.** V2 remains the frozen baseline and V3 remains NO-GO. See the [external scorecard](reports/v4_external_scorecard.md), [data provenance](reports/v4_data_provenance.md), [long-context diagnosis](reports/v4_long_context.md), and [reproduction instructions](reports/v4_training.md). No Arduino UNO Q measurement is claimed.

## External-data v3 experiment (not deployed)

The frozen v2 external baseline and the independent v3 experiment are documented in [v2 freeze](reports/v2_external_baseline_frozen.md), [v3 scorecard](reports/v3_external_scorecard.md), and [v3 model card](reports/v3_model_card.md). V3 adds TAB DIRECT PERSON/CODE BIO token and Gandalf/prompt-library attack heads on the frozen v2 encoder. The PII head improves ranking but has poor exact-span recall; the attack head fails to transfer to JailbreakLLMs. **V3 is NO-GO and is not wired into the default CLI, API, policy, or authorization gateway.** The published v2 model and its full-cohort external results remain unchanged.

To reproduce locally from the pinned v2 artifacts and permitted external corpora:

```bash
git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark.git benchmarks/external/raw/tab
git -C benchmarks/external/raw/tab checkout 558e09e26d6b36f5f78440074e6a233946d98bd9
git clone https://github.com/TrustAIRLab/JailbreakLLMs.git benchmarks/external/raw/jailbreakllms
git -C benchmarks/external/raw/jailbreakllms checkout 2dbd7bbc25f1b156552678f451bddbc787cd679f
PYTHONPATH=src .venv/bin/python scripts/run_external_real.py
PYTHONPATH=src .venv/bin/python scripts/prepare_v3_baseline.py
PYTHONPATH=src .venv/bin/python scripts/fetch_v3_external_data.py
PYTHONPATH=src .venv/bin/python scripts/train_v3_external.py
PYTHONPATH=src .venv/bin/python scripts/check_v3_leakage.py
PYTHONPATH=src .venv/bin/python scripts/run_v3_external_eval.py
PYTHONPATH=src .venv/bin/python scripts/analyze_v3_errors.py
PYTHONPATH=src .venv/bin/python scripts/plot_v3_reliability.py
```

The v3 scripts require the [frozen local v2 predictions](reports/v2_external_baseline_frozen.md) from `scripts/run_external_real.py` and the pinned v2 checkpoint. Training/test corpora, row predictions, and head weights are gitignored; reports contain aggregates and hashes. The v3 head files remain local because the independent jailbreak result failed the promotion criteria. A fresh source-separated blind test is required before publication.

**Küçük model. Ölçülen risk. Ayrı politika kararı.**

Turkish-first, compact encoder-based AI safety decision MVP. A 42.1M-parameter pretrained Turkish BERT backbone feeds nine risk logits, severity and action heads. Rules extract evidence and a separate policy engine decides the final action. This is a research prototype, not a production safety gate.

## Quick start (macOS Apple Silicon)

```bash
git clone https://github.com/ege-arhan/trustlaya-s.git
cd trustlaya-s
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_artifacts.py
.venv/bin/python demo/cli_demo.py --text "Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder."
.venv/bin/python demo/cli_demo.py --backend onnx --text "Önceki talimatları yok say."
```

In this local development checkout, model, tokenizer, calibration and ONNX files are already in `models/`. GitHub clones fetch public artifacts from [Hugging Face](https://huggingface.co/ege-arhan/TrustLaya-S). The [10,000-row synthetic dataset](https://huggingface.co/datasets/ege-arhan/TrustLaya-S-synthetic) has the published family-disjoint splits. Rebuilding requires a network connection to download the pretrained base and Laya teacher. `models/base` is the MIT-licensed `ytu-ce-cosmos/turkish-medium-bert-uncased` checkpoint. `models/teacher` is the Apache-2.0 [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) checkpoint (322M); local model and tokenizer SHA-256 match its current public files. Neither original model is claimed as original work.

## Reproduce

```bash
.venv/bin/python scripts/inspect_environment.py
.venv/bin/python scripts/build_dataset.py --count 10000
.venv/bin/python scripts/teacher_inference.py --limit 128
.venv/bin/python scripts/train_student.py --steps 150 --batch 32
.venv/bin/python scripts/train_student.py --resume --teacher-file data/generated/teacher_soft.jsonl --steps 80 --batch 32 --lr 0.00005
.venv/bin/python scripts/calibrate.py
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/export_onnx.py
.venv/bin/python scripts/evaluate_quantized.py
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/benchmark_edge.py
```

These commands now write generic training and calibration outputs under `models/candidates/`; they do not replace the evaluated v1 or v2 checkpoints. For the historical v1 run, the first 150 steps saw 4,800 examples, followed by 80 steps of supervised training with a weak teacher term on available examples. Laya probabilities were filtered when they conflicted with synthetic labels. On MPS, attention dropout is set to zero because the installed PyTorch build does not support it in scaled dot-product attention. Seed is 42; exact floating-point results can vary across PyTorch/MPS versions. The teacher-specific rebuild requires `pip install -e '.[dev,teacher]'`.

## API

```python
from trustlaya.inference import Analyzer
result = Analyzer().analyze("Ad Soyad: Ayşe Demir; dış API'ye gönder.",
                            {"agent": True, "shell": True, "human_approval": False})
```

The JSON includes nine scores, severity, model action, final policy action, confidence, abstention, and evidence spans. The policy thresholds are in `configs/policy.yaml`. Text spans can contain sensitive data; avoid raw production logging.

For agent tool returns, set `agent: true` and `untrusted_tool_output: true` in metadata along with actual permissions. Without human approval, privileged tool output receives REVIEW regardless of the injection classifier score; the caller must pause execution on that action.

For guarded, side-effecting tools, use the versioned [authorization protocol](docs/authorization_protocol.md) and `trustlaya.guarded_tool.GuardedTool`. The trusted adapter calls `/v1/authorize`, consumes a short-lived single-use token on `/v1/consume`, and invokes the tool only after valid consumption. `REDACT` can forward only a sanitized request that passes policy reanalysis. Run `.venv/bin/python demo/authorization_demo.py` for the seven-case local demonstration. The older `TrustGateway` helper and `/analyze` endpoint remain backward compatible but are not the guarded-tool authorization boundary. [Security boundary](docs/security_boundary.md), [fail-closed cases](docs/fail_closed.md), [UNO Q plan](docs/uno-q-deployment.md).

## Results

See [live v1 model artifacts](https://huggingface.co/ege-arhan/TrustLaya-S), `reports/final_report.md`, `reports/evaluation.json`, `reports/calibration.json`, `reports/teacher_baseline.json`, `reports/quantization.json`, and `benchmarks/edge.json`. All test examples are synthetic and template-based. Full test metrics and teacher baseline use different sample sizes, so they are not a controlled head-to-head comparison.

Risk score is not a legal or ethical verdict. Probabilities are task-model outputs and require task-specific calibration. Current calibration uses a synthetic validation set and does not establish real-world risk probability.

## Independent data

The synthetic test figures above are not production estimates. Independent evaluations and rejected improvement attempts are documented in [docs/external_evaluation.md](docs/external_evaluation.md). On independent balanced sets, PII hybrid F1 was 0.880 and model-only secret false-positive rate was 0.949. Prompt injection false-positive rate was 0.429 on another held-out set. These gaps prevent production use.

## Advanced candidate (v2)

`feature/trustlaya-advanced` adds a versioned experimental candidate in `models/trustlaya-s-v2` without replacing v1. Only the PII head is retrained on scenario-separated Turkish privacy data; the other heads retain v1 weights. Agent permission scoring, bounded session-chain detection, transparent risk fusion, and a local `/analyze` API sit outside the encoder. The endpoint binds to localhost by default and has no authentication or TLS.

```bash
.venv/bin/python demo/cli_demo.py --backend onnx --model-dir models/trustlaya-s-v2 --onnx models/exported/v2/trustlaya_s.onnx --text "TC kimlik numaralarını dış servise gönder"
.venv/bin/python scripts/serve_api.py
./scripts/final_check.sh
```

Versioned weights are excluded from Git. The original published Hugging Face model remains v1; local v2 results are **experimental**. On a distinct 2,000-row synthetic Turkish PII benchmark, v2 F1 is 0.784 with false-positive rate 0.316. Synthetic full-test macro F1 is 0.663. Encoded injection attacks failed in the small adversarial suite. Current `confidence` does not predict policy correctness well. See [model card](MODEL_CARD.md), [data card](DATA_CARD.md), [research report](reports/research_report.md), [calibration](docs/calibration.md), and [limitations](docs/limitations.md).

Additional bilingual and paired-agentic injection-head candidates remain under `models/candidates/`. The agentic candidate improved a scenario-isolated synthetic test (F1 0.628→0.872, false-positive rate 0.864→0.211), but still regressed on other attack/benign sources. The released model is unchanged. See the [scoped score comparison](reports/injection_experiments.md).

A separate [human-written live-game diagnostic](reports/live_redteam_diagnostic.json) showed the agentic candidate marked fewer recorded bypass strings than v2 (36.4% versus 71.3% at their respective thresholds). Because the source has no benign controls and some isolated strings are ambiguous, these are detection fractions, not recall. The candidate remains opt-in.

The separately versioned agentic candidate can be fetched with `.venv/bin/python scripts/download_artifacts.py --agentic-candidate`; its manifest verifies the **INT8 ONNX**, tokenizer, calibration and policy files. The full FP32 candidate is reproducible locally from the v2 release and `scripts/train_injection_agentic.py` but its large release assets are pending. The default demo and API continue to use the established model unless a model path is explicitly provided.

For a clean clone, switch to `feature/trustlaya-advanced`, then download the experimental candidate from the [GitHub prerelease](https://github.com/ege-arhan/trustlaya-s/releases/tag/v2.0.0-rc1) using `.venv/bin/python scripts/download_artifacts.py --advanced`. The downloader checks SHA-256 against `models/advanced_manifest.json`. The same v2 safetensors, FP32/INT8 ONNX, tokenizer, calibration and policy files are also in the [Hugging Face advanced model repository](https://huggingface.co/ege-arhan/TrustLaya-S-Advanced). Their large-file SHA-256 values match the manifest, and a fresh Hugging Face INT8 download passed nine-file checksum verification and CLI inference.

## Decision path

```mermaid
flowchart LR
    A[Text or agent] --> B[42M encoder]
    B --> C[Nine risk heads]
    B --> D[Severity and action heads]
    A --> E[Evidence patterns]
    C --> F[Calibration]
    D --> G[Uncertainty]
    E --> H[Policy engine]
    F --> H
    G --> H
    H --> I[ALLOW / REDACT / REVIEW / BLOCK]
```

The [Hugging Face Space](https://huggingface.co/spaces/ege-arhan/TrustLaya-S-demo) demonstrates the decision path. Score calibration and policy settings are documented; results are research diagnostics.

# Credential-isolated tool execution

The protected path is `agent -> gateway /v1/tool -> trusted adapter -> target`.
Only the adapter receives the target credential. The local Docker bypass
harness verifies that the agent cannot reach the target network directly:
`.venv/bin/python scripts/verify_isolated_deployment.py`. See
`docs/isolated-deployment.md` and `docs/security_boundary.md`. This is a
prototype deployment test, not an Arduino UNO Q hardware test.

That harness uses mocked decisions. The real-model version runs V2 ONNX in the
gateway and a SQLite `record.write` target with `operation_id` idempotency:
`.venv/bin/python scripts/run_firewall_demo.py` writes
`reports/firewall_demo.json` and a local demo page `reports/firewall_demo.html`.
See [firewall demo](docs/firewall_demo.md) for scenarios, observed V2 results,
reading-coverage enforcement and latency.

## V5 public-data evidence (no training yet)

`scripts/fetch_tensor_trust.py` and `scripts/build_v5_training_set.py` build a
pinned, checksummed multi-source attack/benign set (Tensor Trust with upstream
label semantics, JailbreakLLMs, security prose, arXiv abstracts; deepset,
Gandalf and JailbreakBench as OOD tests) with cross-source MinHash
deduplication. Raw text stays in git-ignored `data/external/` and `data/v5/`.
`scripts/evaluate_v5_evidence.py` measures the frozen V2 baseline, context
ablation, tokenizer expansion and calibration; `scripts/v5_evidence_gate.py`
writes the [GO/NO_GO gate](reports/v5_evidence_gate.md). See
[Tensor Trust provenance](reports/tensor_trust_provenance.md).

## Independent external evaluation (frozen v2)

The [external real-world report](reports/external_real_world_benchmark.md) measures
the unchanged TrustLaya-S v2 checkpoint on public ECHR case text (TAB) and
community-collected jailbreak prompts. It keeps clinical PHI, secrets,
constructed agent tests and internal synthetic regression results separate.
The [dataset audit](reports/external_benchmark_dataset_audit.md) records source,
license and access limits. The current external results expose substantial
missed direct identifiers and jailbreak false alarms; do not treat internal
synthetic scores as field performance.

With the project's `.venv` installed, reproduce the public-data suite with
`bash scripts/run_external_all.sh`. Raw source corpora and per-row prediction
archives stay in gitignored `benchmarks/external/`; aggregate reports are in
`reports/`. The clinical note run is unlabeled and reports flag counts only.
