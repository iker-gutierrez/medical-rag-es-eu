#!/bin/bash
#SBATCH --job-name=q35-4b-notk-gen
#SBATCH --array=0-21%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_4b_no_think_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_4b_no_think_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job q35-4b-notk-gen started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

CONFIGS=(
  configs/experiments/1172_qwen35_4b_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1173_qwen35_4b_rag_e5_topk1_no_think_extractive_mixed_dev.json
  configs/experiments/1174_qwen35_4b_rag_e5_topk3_no_think_extractive_mixed_dev.json
  configs/experiments/1175_qwen35_4b_rag_e5_topk5_no_think_extractive_mixed_dev.json
  configs/experiments/1176_qwen35_4b_rag_e5_rerank1_no_think_extractive_mixed_dev.json
  configs/experiments/1177_qwen35_4b_rag_e5_rerank3_no_think_extractive_mixed_dev.json
  configs/experiments/1178_qwen35_4b_rag_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1179_qwen35_4b_3shot_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1180_qwen35_4b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1181_qwen35_4b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1182_qwen35_4b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1194_qwen35_4b_no_rag_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1195_qwen35_4b_rag_e5_topk1_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1196_qwen35_4b_rag_e5_topk3_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1197_qwen35_4b_rag_e5_topk5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1198_qwen35_4b_rag_e5_rerank1_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1199_qwen35_4b_rag_e5_rerank3_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1200_qwen35_4b_rag_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1201_qwen35_4b_3shot_no_rag_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1202_qwen35_4b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1203_qwen35_4b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1204_qwen35_4b_rag_3shot_e5_rerank5_no_think_extractive_mixed_sf_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job q35-4b-notk-gen finished at $(date)"
