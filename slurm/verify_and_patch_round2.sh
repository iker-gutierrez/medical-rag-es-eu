#!/bin/bash
#SBATCH --job-name=verify-patch-r2
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=8GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/verify_patch_r2_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/verify_patch_r2_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

echo "verify-patch-r2 started on $(hostname) at $(date)"

echo ""
echo "=== Verifying round-2 reruns ==="
set +e
python scripts/verify_rerun_truncation.py \
  --manifest /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_round2.json \
  --config-key round2_config \
  --output-key round2_output
VERIFY_EXIT=$?
set -e

if [ $VERIFY_EXIT -ne 0 ]; then
  echo ""
  echo "Round-2 verification reported issues (exit $VERIFY_EXIT). NOT patching automatically."
  echo "Inspect the report above -- these 9 examples are genuine long-tail outliers;"
  echo "if still truncated at 16000, they may need per-example inspection rather than"
  echo "another blind token bump."
  exit $VERIFY_EXIT
fi

echo ""
echo "=== Round 2 clean. Patching round 1 (bulk fixes) then round 2 (stragglers) ==="
python scripts/patch_truncated_predictions.py \
  --manifest /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest.json \
  --output-key rerun_output

python scripts/patch_truncated_predictions.py \
  --manifest /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_round2.json \
  --output-key round2_output

echo ""
echo "verify-patch-r2 finished at $(date)"
