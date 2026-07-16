#!/bin/bash
# Waits for calibration job 9769 (calib_marag_eu_latxa_topk1 + calib_scot_eu_latxa_topk1)
# to finish, applies the calibrated MA-RAG threshold to config 1323, then submits
# the Basque Latxa e5-top1 reasoning pipeline runs: the 3 non-MA-RAG pipelines
# immediately, and MA-RAG once its threshold is confirmed set (not null).
set -uo pipefail
cd /home/igutierrez134/med_rag_thesis
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
LOG=experiments/slurm_logs/chain_reasoning_eu_latxa_topk1.log
exec >>"$LOG" 2>&1

echo "=== chain_reasoning_eu_latxa_topk1 started $(date) ==="

while squeue -j 9769 -h 2>/dev/null | grep -q .; do
  sleep 20
done
echo "job 9769 (calibration) done at $(date)"

echo "=== analyzing calibration + applying threshold $(date) ==="
python scripts/analyze_conflict_calibration.py
python scripts/set_calibrated_thresholds.py

# Sanity check: 1323's threshold must no longer be null before its array is
# submitted, or run_reasoning_pipeline.py will crash on float(None) for all 3 seeds.
python3 - <<'EOF'
import json, sys
c = json.load(open("configs/experiments/1323_latxa_llama31_8b_marag_e5_topk1_extractive_mixed_eu_dev.json"))
if c.get("conflict_threshold_open") is None:
    print("ERROR: conflict_threshold_open still null after set_calibrated_thresholds.py -- "
          "calibration run likely produced no usable data. Not submitting MA-RAG array.")
    sys.exit(1)
print(f"OK: conflict_threshold_open = {c['conflict_threshold_open']}")
EOF
THRESHOLD_OK=$?

echo "=== submitting 3-config Latxa top-1 array (structured_cot, thought_rag, thought_rag_iter) $(date) ==="
sbatch slurm/reasoning_eu_latxa_topk1_seeded.sh

if [ "$THRESHOLD_OK" -eq 0 ]; then
  echo "=== submitting MA-RAG Latxa top-1 array $(date) ==="
  sbatch slurm/reasoning_eu_latxa_marag_seeded.sh
else
  echo "=== SKIPPED MA-RAG array: threshold not set, needs manual attention ==="
fi

echo "=== chain_reasoning_eu_latxa_topk1 finished $(date) ==="
