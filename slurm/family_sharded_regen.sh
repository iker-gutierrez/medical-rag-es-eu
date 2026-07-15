#!/bin/bash
#SBATCH --job-name=family-sharded-regen
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/family_sharded_regen_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/family_sharded_regen_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# One model family per submission (FAMILY env var), its configs split across
# the two array tasks (shard 0 and 1) running concurrently on both GPU slots.
# Families are chained sequentially via --dependency. This way every family
# is measured under the same condition -- 2-GPU parallelism with a same-model
# neighbor -- keeping per-sample cost/timing metrics comparable across models.

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

: "${FAMILY:?Set FAMILY via sbatch --export=ALL,FAMILY=<family>}"
SPLIT="${SPLIT:-mixed}"

echo "family-sharded-regen family=$FAMILY split=$SPLIT shard=${SLURM_ARRAY_TASK_ID}/2 started on $(hostname) at $(date)"
nvidia-smi || true

python scripts/run_casionly_regeneration.py \
  --family "$FAMILY" --split "$SPLIT" --shard "${SLURM_ARRAY_TASK_ID}/2"

echo "family-sharded-regen family=$FAMILY shard=${SLURM_ARRAY_TASK_ID} finished at $(date)"
