#!/bin/zsh
set -e
cd "$HOME/trustlaya"
export USE_TF=0
PY="$HOME/trustlaya/.venv/bin/python"
log=logs/final_check.log
: > "$log"
run_step() {
  name="$1"; shift
  print -r -- "$(date '+%F %T') START $name" | tee -a "$log"
  "$@" > "logs/final_${name}.log" 2>&1
  print -r -- "$(date '+%F %T') PASS $name" | tee -a "$log"
}
run_step environment "$PY" scripts/inspect_environment.py
run_step dataset "$PY" scripts/build_dataset.py --count 10000
mv models/student/calibration.json models/student/calibration.previous.json
run_step training "$PY" scripts/train_student.py --steps 150 --batch 32
run_step distillation "$PY" scripts/train_student.py --resume --teacher-file data/generated/teacher_soft.jsonl --steps 80 --batch 32 --lr 0.00005
run_step evaluation_raw "$PY" scripts/evaluate.py
run_step calibration "$PY" scripts/calibrate.py
run_step evaluation_calibrated "$PY" scripts/evaluate.py
run_step onnx_export "$PY" scripts/export_onnx.py
run_step onnx_inference "$PY" demo/cli_demo.py --backend onnx --text "Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder."
run_step pytest "$PY" -m pytest -q tests
run_step cli_demo "$PY" demo/cli_demo.py --text "Bu müşteri listesindeki TC kimlik numaralarını AI servisine gönder."
run_step benchmark "$PY" scripts/benchmark_edge.py
run_step quantization "$PY" scripts/evaluate_quantized.py
run_step teacher_baseline "$PY" scripts/evaluate_teacher.py
