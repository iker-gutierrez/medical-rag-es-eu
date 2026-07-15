#!/bin/bash
#SBATCH --job-name=patch-trunc-retry
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=8GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/patch_truncation_retry_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/patch_truncation_retry_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# CPU-only, no GPU needed -- merges the targeted retry-regeneration output
# back into each original run's predictions.jsonl.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate

echo "patch-truncation-retry started at $(date)"
python scripts/patch_truncation_retry_results.py
echo "patch-truncation-retry finished at $(date)"
