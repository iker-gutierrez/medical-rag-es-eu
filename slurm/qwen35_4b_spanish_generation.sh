#!/bin/bash
#SBATCH --job-name=qwen35-4b-es-gen
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_4b_spanish_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_4b_spanish_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

CONFIGS=(
  configs/experiments/94_qwen35_4b_no_rag_no_think_extractive_sns1064_dev.json
  configs/experiments/95_qwen35_4b_no_rag_think_extractive_sns1064_dev.json
  configs/experiments/96_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_sns1064_dev.json
  configs/experiments/97_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_sns1064_dev.json
  configs/experiments/98_qwen35_4b_no_rag_no_think_extractive_casimedicos_dev.json
  configs/experiments/99_qwen35_4b_no_rag_think_extractive_casimedicos_dev.json
  configs/experiments/100_qwen35_4b_rag_no_think_e5_topk3_extractive_casimedicos_dev.json
  configs/experiments/101_qwen35_4b_rag_think_e5_topk3_extractive_casimedicos_dev.json
  configs/experiments/102_qwen35_4b_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/103_qwen35_4b_no_rag_think_extractive_mixed_dev.json
  configs/experiments/104_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_mixed_dev.json
  configs/experiments/105_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_mixed_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Qwen3.5-4B Spanish generation started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Qwen3.5-4B Spanish generation finished at $(date)"
