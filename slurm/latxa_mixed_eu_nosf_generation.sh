#!/bin/bash
#SBATCH --job-name=latxa-eu-nosf
#SBATCH --array=0-10%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/latxa_mixed_eu_nosf_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/latxa_mixed_eu_nosf_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job latxa-eu-nosf started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

CONFIGS=(
  configs/experiments/1051_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev.json
  configs/experiments/1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev.json
  configs/experiments/1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev.json
  configs/experiments/1054_latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev.json
  configs/experiments/1055_latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev.json
  configs/experiments/1056_latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev.json
  configs/experiments/1057_latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev.json
  configs/experiments/1058_latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev.json
  configs/experiments/1059_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev.json
  configs/experiments/1060_latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev.json
  configs/experiments/1061_latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job latxa-eu-nosf finished at $(date)"
