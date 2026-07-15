#!/bin/bash
#SBATCH --job-name=retry-truncated
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/retry_truncated_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/retry_truncated_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "retry-truncated task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
nvidia-smi || true

LABELS=(
  "Mistral-7B ES"
  "Qwen3.5-9B no_think ES"
  "Llama-3.1-8B EU"
  "Latxa-8B EU"
)

LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"
echo "Label: $LABEL"

python scripts/run_truncation_reruns.py --label "$LABEL"

echo "retry-truncated finished at $(date)"
