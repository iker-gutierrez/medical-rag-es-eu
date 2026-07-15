#!/bin/bash
#SBATCH --job-name=q35-9b-gen
#SBATCH --array=0-10%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_no_think_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_no_think_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job q35-9b-gen started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

CONFIGS=(
  configs/experiments/1128_qwen35_9b_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1129_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev.json
  configs/experiments/1130_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev.json
  configs/experiments/1131_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev.json
  configs/experiments/1132_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev.json
  configs/experiments/1133_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev.json
  configs/experiments/1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1135_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1136_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1137_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1138_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

SEED_ARG=""
[ -n "${SEED:-}" ] && SEED_ARG="--seed ${SEED}"
python scripts/run_generation_from_config.py --config "$CONFIG" ${SEED_ARG}

echo "Job q35-9b-gen finished at $(date)"
