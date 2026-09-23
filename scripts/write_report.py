import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
def f(x):return f'{x:.3f}'
def mib(n):return f'{n/2**20:.1f} MiB'
def main():
    train=read('reports/training.json');data=read('data/processed/validation.json');ev=read('reports/evaluation.json');cal=read('reports/calibration.json');quant=read('reports/quantization.json');size=read('reports/model_sizes.json');bench=read('benchmarks/edge.json');baseline=read('reports/teacher_baseline.json')
    mean_ece=sum(x['calibrated']['ece'] for x in cal.values())/len(cal)
    mean_brier=sum(x['calibrated']['brier'] for x in cal.values())/len(cal)
    raw_ece=sum(x['raw']['ece'] for x in cal.values())/len(cal)
    raw_brier=sum(x['raw']['brier'] for x in cal.values())/len(cal)
    raw_nll=sum(x['raw']['nll'] for x in cal.values())/len(cal)
    cal_nll=sum(x['calibrated']['nll'] for x in cal.values())/len(cal)
    per_task='\n'.join(f"| {k} | {f(v['f1'])} | {f(v['recall'])} | {f(v['false_positive_rate'])} | {f(v['false_negative_rate'])} | {v['support']} |" for k,v in ev['tasks'].items())
    models=[('teacher','322M',baseline['latency_p50_ms']['teacher_cpu'],mib((R/'models/teacher/model.safetensors').stat().st_size)),('student',f"{train['parameters']/1e6:.1f}M",baseline['latency_p50_ms']['student_mps'],mib(size['student_fp32_bytes'])),('rules','0',None,'0')]
    comparison='\n'.join(f"| {name} | {params} | {f(baseline[name]['macro_f1'])} | {f(baseline[name]['macro_recall'])} | {f(baseline[name]['mean_ece'])} | {f(lat) if lat is not None else 'not measured'} | {artifact} |" for name,params,lat,artifact in models)
    txt=f'''# TrustLaya-S final report

## Problem and goal

Compact edge-oriented AI safety decision support for Turkish-first text and AI agent permission metadata. This is a working MVP, not a production safety gate.

## Architecture and student

Pretrained Turkish BERT shared encoder, mean pooling, nine independent binary risk heads, severity head, and advisory action head. Total parameters: **{train['parameters']:,}**; trainable during this run: **{train['trainable_parameters']:,}**. Embeddings and first two encoder layers were frozen. 96-token input. Source backbone: [YTU CE Cosmos Turkish Medium BERT](https://huggingface.co/ytu-ce-cosmos/turkish-medium-bert-uncased), MIT license. Model output is calibrated per task, combined with deterministic evidence, then sent to an independent policy engine. The policy emits ALLOW, REDACT, REVIEW or BLOCK.

## Teacher and distillation

[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), multilingual checkpoint, Apache-2.0 license; approximately 322M parameters. 128 training rows were queried for nine typed `noul` probabilities. There were {train['teacher_examples']} unique text keys in the teacher file. Student loss is binary cross-entropy for risk heads plus 0.3 severity cross-entropy plus 0.3 action cross-entropy; the weak teacher binary cross-entropy term has weight 0.05 only when teacher's binary side agrees with synthetic label. Teacher predictions are not ground truth. Training: 150 supervised steps and 80 resumed weak-distillation steps, batch 32. Logs: `logs/final_training.log` and `logs/final_distillation.log`.

## Dataset

10,000 synthetic samples, eleven categories, Turkish/English/mixed templates. Train {data['counts']['train']}, validation {data['counts']['val']}, test {data['counts']['test']}. Template families and exact texts do not cross splits. Synthetic examples are controlled, but many examples share a small set of templates; results are not evidence of open-world robustness.

## Full held-out synthetic test ({ev['examples']} examples)

Mean task accuracy: **{f(ev['accuracy'])}**; macro F1: **{f(ev['macro_f1'])}**. Threshold is 0.5 after calibration and deterministic PII/secret score floors.

| Task | F1 | Recall | False positive rate | False negative rate | Positive support |
|---|---:|---:|---:|---:|---:|
{per_task}

## Baselines on same 128-row test subset

The following scores share the same 128 examples. Teacher probabilities were not domain-calibrated. Rule-only baseline covers PII and secrets; other tasks predict zero. F1 and recall are macro means over nine tasks. Latency is batch 1, measured in the baseline script, with student on MPS and teacher on CPU.

| Model | Parameters | Macro F1 | Macro recall | Mean ECE | p50 latency ms | Weight size |
|---|---:|---:|---:|---:|---:|---:|
{comparison}

The full 1,975-row test and the 128-row baseline subset must not be compared as if they were the same experiment. Rule-based PII/secret baseline F1 on the full test is {f(ev['baseline_rules']['pii']['f1'])}/{f(ev['baseline_rules']['secret']['f1'])}, which reflects exact synthetic patterns.

## Calibration and uncertainty

Nine independent temperature scalars fitted on validation. Mean ECE raw **{f(raw_ece)}**, calibrated **{f(mean_ece)}**. Mean Brier raw **{f(raw_brier)}**, calibrated **{f(mean_brier)}**. Mean NLL raw **{f(raw_nll)}**, calibrated **{f(cal_nll)}**. Per-task raw and calibrated results are in `reports/calibration.json`. Calibration improved mean metrics but worsened ECE for some individual tasks. Confidence is the minimum of severity and action softmax peaks, a decisiveness heuristic rather than calibrated correctness. When confidence is below policy threshold 0.60, policy returns REVIEW and `abstain=true` unless deterministic evidence takes precedence. A 0.90 risk at 0.95 confidence can trigger BLOCK; a 0.90 risk at 0.55 confidence triggers REVIEW.

## Evidence and policy

Regex/pattern spans cover Turkish national ID checksum, phone, email, validated IP, Luhn card, IBAN, labeled address/name/customer ID, API keys, bearer tokens, passwords, database credentials and related secrets. PII external transfer triggers REDACT; secret evidence triggers BLOCK. Agent shell or credential access without human approval triggers REVIEW. Thresholds live in `configs/policy.yaml`. Model `model_action` remains advisory; final `action` is policy output.

## Export, size and latency

| Artifact | Size |
|---|---:|
| Student FP32 safetensors | {mib(size['student_fp32_bytes'])} |
| Student FP16 safetensors | {mib(size['student_fp16_bytes'])} |
| ONNX FP32 | {mib(size['onnx_fp32_bytes'])} |
| ONNX INT8 per-channel | {mib(size['onnx_int8_bytes'])} |

| Backend | Cold start ms | Warm p50 ms, batch 1 | Approx RSS delta MiB |
|---|---:|---:|---:|
'''
    for name,values in bench.items():txt+=f"| {name} | {f(values['cold_start_ms'])} | {f(values['warm_p50_ms'])} | {f(values['rss_delta_mib_approx'])} |\n"
    txt+=f'''
RSS deltas were measured sequentially in one process; allocator reuse and already loaded models make them approximate and unsuitable for direct backend comparison. INT8 changed **{f(quant['action_disagreement']*100)}%** of final actions versus FP32 on {quant['examples']} synthetic test rows. INT8 is experimental and should not be deployed without a stricter parity gate. Detailed F1 differences are in `reports/quantization.json`.

## Test and verification

`pytest`: 11 passed. ONNX checker and ONNX CPU inference passed. CLI produced REDACT for the requested Turkish PII transfer example. Stage outcomes are in `logs/final_check.log`. No Arduino UNO Q board was connected or measured.

## Limitations and next work

Risk score is not a legal or ethical verdict. Probabilities are task-model outputs and require task-specific calibration. Data governance F1 is zero on the held-out synthetic templates. Dangerous-instruction and ethics false negative rates are high. Teacher is weak on the specialized schema. Template-generated samples overstate pattern performance and do not cover natural incidents, adversarial paraphrases, or shifted populations. Confidence is not calibrated correctness. English support is limited by a Turkish-first backbone. Next priority: independently labeled real Turkish/English validation and test sets, hard negative collection, then threshold and calibration refit. Only after that, revisit INT8 and UNO Q deployment.
'''
    (R/'reports/final_report.md').write_text(txt)
    print('wrote',R/'reports/final_report.md')
if __name__=='__main__':main()
