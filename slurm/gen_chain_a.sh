#!/bin/bash
#SBATCH --job-name=medrag-chain-a
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/chain_a_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/chain_a_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

CONFIGS=(
  "configs/experiments/20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev.json"
  "configs/experiments/29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev.json"
  "configs/experiments/31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev.json"
  "configs/experiments/33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev.json"
  "configs/experiments/37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job finished at $(date)"
