#!/bin/bash
#SBATCH --job-name=verify-patch-trunc
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=8GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/verify_patch_trunc_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/verify_patch_trunc_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

echo "verify-patch-trunc started on $(hostname) at $(date)"

echo ""
echo "=== Verifying rerun truncation ==="
set +e
python scripts/verify_rerun_truncation.py
VERIFY_EXIT=$?
set -e

if [ $VERIFY_EXIT -ne 0 ]; then
  echo ""
  echo "Verification reported issues (exit $VERIFY_EXIT). NOT patching automatically."
  echo "Inspect the report above, decide on next token limit if still truncated, then"
  echo "rerun manually: python scripts/patch_truncated_predictions.py"
  exit $VERIFY_EXIT
fi

echo ""
echo "=== All retries clean. Patching predictions ==="
python scripts/patch_truncated_predictions.py

echo ""
echo "verify-patch-trunc finished at $(date)"
