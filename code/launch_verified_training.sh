#!/bin/bash
set -eu
BASE=/path/to/project_root/submission_20261026_v4/P1
PY=python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1
test -f "$BASE/private/READY.json"
"$PY" -u "$BASE/reproducibility/verify_corrected.py" --stage data > "$BASE/logs/data_integrity.log" 2>&1
run_queue() {
  gpu="$1"
  scheme="$2"
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$BASE/reproducibility/train_corrected.py" --scheme "$scheme" --seed "$seed" > "$BASE/logs/${scheme}_seed${seed}.log" 2>&1
    echo "COMPLETED scheme=$scheme seed=$seed"
  done
}
run_queue 0 tree &
TREE=$!
run_queue 1 zone &
ZONE=$!
wait "$TREE"
wait "$ZONE"
"$PY" -u "$BASE/reproducibility/verify_corrected.py" --stage full > "$BASE/logs/full_integrity.log" 2>&1
"$PY" -u "$BASE/reproducibility/analyze_corrected.py" > "$BASE/logs/analysis_corrected.log" 2>&1
"$PY" -u "$BASE/reproducibility/render_figures_from_tables.py" > "$BASE/logs/figures.log" 2>&1
echo 'ALL SIX CORRECTED MODELS, VALIDATION, ANALYSES AND FIGURES COMPLETE'
