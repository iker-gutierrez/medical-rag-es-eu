# Waits for the ES no-think reasoning pipeline arrays (9787 structured_cot/
# thought_rag/thought_rag_iter, 9788 marag) to finish, then evaluates both the
# ES no-think (1330-1333) and EU Latxa top-1 (1320-1323, already finished
# earlier tonight but never evaluated) batches, then regenerates
# reports/metrics/reasoning_pipeline_dev_results.md with the corrected row
# definitions (write_reasoning_pipeline_summary.py, already updated).
set -uo pipefail
cd /home/igutierrez134/med_rag_thesis
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
LOG=experiments/slurm_logs/chain_reasoning_summary_update.log
exec >>"$LOG" 2>&1

echo "=== chain_reasoning_summary_update started $(date) ==="

for job in 9787 9788; do
  while squeue -j "$job" -h 2>/dev/null | grep -q .; do
    sleep 20
  done
  echo "job ${job} done at $(date)"
done

echo "=== submitting evaluation (ES no-think + EU Latxa top-1) $(date) ==="
EVAL_JOB=$(sbatch --parsable slurm/eval_reasoning_pipelines_v2.sh)
echo "submitted eval job ${EVAL_JOB}"

while squeue -j "$EVAL_JOB" -h 2>/dev/null | grep -q .; do
  sleep 20
done
echo "job ${EVAL_JOB} (evaluation) done at $(date)"

echo "=== regenerating reasoning_pipeline_dev_results.md $(date) ==="
python scripts/write_reasoning_pipeline_summary.py

echo "=== chain_reasoning_summary_update finished $(date) ==="
