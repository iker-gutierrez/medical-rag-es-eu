#!/bin/bash
#SBATCH --job-name=qwen-es-think
#SBATCH --array=0-10
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_es_think_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_es_think_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "qwen-es-think task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"
nvidia-smi || true

CONFIGS=(
  configs/experiments/1270_qwen35_9b_no_rag_think_extractive_mixed_dev.json
  configs/experiments/1271_qwen35_9b_rag_e5_topk1_think_extractive_mixed_dev.json
  configs/experiments/1272_qwen35_9b_rag_e5_topk3_think_extractive_mixed_dev.json
  configs/experiments/1273_qwen35_9b_rag_e5_topk5_think_extractive_mixed_dev.json
  configs/experiments/1274_qwen35_9b_rag_e5_rerank1_think_extractive_mixed_dev.json
  configs/experiments/1275_qwen35_9b_rag_e5_rerank3_think_extractive_mixed_dev.json
  configs/experiments/1276_qwen35_9b_rag_e5_rerank5_think_extractive_mixed_dev.json
  configs/experiments/1277_qwen35_9b_3shot_no_rag_think_extractive_mixed_dev.json
  configs/experiments/1278_qwen35_9b_rag_sns1064_e5_rerank5_think_extractive_mixed_dev.json
  configs/experiments/1279_qwen35_9b_rag_casimedicos_e5_rerank5_think_extractive_mixed_dev.json
  configs/experiments/1280_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Config: $CONFIG"

for SEED in 42 43 44; do
  echo "--- Seed $SEED ---"
  python scripts/run_generation_from_config.py --config "$CONFIG" --seed "$SEED"
done

echo "Generation done at $(date)"
