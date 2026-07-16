#!/bin/bash
# Submits the ES no-think calibration, waits for it, applies the calibrated
# MA-RAG threshold to config 1333, then submits the Spanish Qwen no-think +
# rerank5 reasoning pipeline runs: the 3 non-MA-RAG pipelines immediately, and
# MA-RAG once its threshold is confirmed set (not null).
set -uo pipefail
cd /home/igutierrez134/med_rag_thesis
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
LOG=experiments/slurm_logs/chain_reasoning_es_nothink.log
exec >>"$LOG" 2>&1

echo "=== chain_reasoning_es_nothink started $(date) ==="

CALIB_JOB=$(sbatch --parsable slurm/reasoning_calibrate_es_nothink.sh)
echo "submitted calibration job ${CALIB_JOB} at $(date)"

while squeue -j "$CALIB_JOB" -h 2>/dev/null | grep -q .; do
  sleep 20
done
echo "job ${CALIB_JOB} (calibration) done at $(date)"

echo "=== analyzing calibration + applying threshold $(date) ==="
python scripts/analyze_conflict_calibration.py
python scripts/set_calibrated_thresholds.py

python3 - <<'EOF'
import json, sys
c = json.load(open("configs/experiments/1333_qwen35_9b_marag_e5_rerank5_no_think_extractive_mixed_dev.json"))
if c.get("conflict_threshold_open") is None:
    print("ERROR: conflict_threshold_open still null after set_calibrated_thresholds.py -- "
          "calibration run likely produced no usable data. Not submitting MA-RAG array.")
    sys.exit(1)
print(f"OK: conflict_threshold_open = {c['conflict_threshold_open']}")
EOF
THRESHOLD_OK=$?

echo "=== submitting 3-config ES no-think array (structured_cot, thought_rag, thought_rag_iter) $(date) ==="
sbatch slurm/reasoning_es_nothink_seeded.sh

if [ "$THRESHOLD_OK" -eq 0 ]; then
  echo "=== submitting MA-RAG ES no-think array $(date) ==="
  sbatch slurm/reasoning_es_nothink_marag_seeded.sh
else
  echo "=== SKIPPED MA-RAG array: threshold not set, needs manual attention ==="
fi

echo "=== chain_reasoning_es_nothink finished $(date) ==="
