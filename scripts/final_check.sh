#!/bin/zsh
set -e
cd "${0:A:h:h}"
export USE_TF=0
PY="${PWD}/.venv/bin/python"
mkdir -p logs
log=logs/advanced_check.log
: > "$log"
run_step() {
  name="$1"; shift
  print -r -- "$(date '+%F %T') START $name" | tee -a "$log"
  "$@" > "logs/advanced_${name}.log" 2>&1
  print -r -- "$(date '+%F %T') PASS $name" | tee -a "$log"
}
run_step environment "$PY" scripts/inspect_environment.py
run_step dataset_audit "$PY" scripts/audit_dataset.py
run_step training_smoke "$PY" scripts/train_student.py --steps 1 --batch 4
run_step evaluation "$PY" scripts/evaluate_advanced.py
run_step recalibration_candidate "$PY" scripts/calibrate.py
run_step adversarial "$PY" scripts/evaluate_adversarial.py
run_step onnx_export "$PY" scripts/export_onnx.py
run_step onnx_parity "$PY" scripts/evaluate_onnx_v2.py
run_step benchmark "$PY" scripts/benchmark.py
run_step reliability "$PY" scripts/plot_reliability.py
run_step model_health "$PY" scripts/model_health.py
run_step pytest "$PY" -m pytest -q tests
run_step cli_demo "$PY" demo/cli_demo.py --backend onnx --model-dir models/trustlaya-s-v2 --onnx models/exported/v2/trustlaya_s.onnx --text "Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder."
