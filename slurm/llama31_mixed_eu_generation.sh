#!/bin/bash
#SBATCH --job-name=llama-eu
#SBATCH --array=0-10%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llama31_mixed_eu_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llama31_mixed_eu_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job llama-eu started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

CONFIGS=(
  configs/experiments/1040_llama31_8b_no_rag_extractive_mixed_eu_dev.json
  configs/experiments/1041_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev.json
  configs/experiments/1042_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev.json
  configs/experiments/1043_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev.json
  configs/experiments/1044_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev.json
  configs/experiments/1045_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev.json
  configs/experiments/1046_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev.json
  configs/experiments/1047_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev.json
  configs/experiments/1048_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev.json
  configs/experiments/1049_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev.json
  configs/experiments/1050_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

SEED_ARG=""
[ -n "${SEED:-}" ] && SEED_ARG="--seed ${SEED}"
python scripts/run_generation_from_config.py --config "$CONFIG" ${SEED_ARG}

echo "Job llama-eu finished at $(date)"
