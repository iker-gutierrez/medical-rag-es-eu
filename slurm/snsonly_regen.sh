#!/bin/bash
#SBATCH --job-name=snsonly-regen
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/snsonly_regen_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/snsonly_regen_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "snsonly-regen task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
nvidia-smi || true

FAMILIES=(qwen-nothink qwen-think mistral llama latxa)
FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
echo "Family: $FAMILY"

python scripts/run_casionly_regeneration.py --family "$FAMILY" --split sns1064

echo "snsonly-regen task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
