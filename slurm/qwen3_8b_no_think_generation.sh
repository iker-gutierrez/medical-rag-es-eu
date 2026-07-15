#!/bin/bash
#SBATCH --job-name=q3-8b-notk-gen
#SBATCH --array=0-21%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen3_8b_no_think_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen3_8b_no_think_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job q3-8b-notk-gen started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

CONFIGS=(
  configs/experiments/1216_qwen3_8b_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1217_qwen3_8b_rag_e5_topk1_no_think_extractive_mixed_dev.json
  configs/experiments/1218_qwen3_8b_rag_e5_topk3_no_think_extractive_mixed_dev.json
  configs/experiments/1219_qwen3_8b_rag_e5_topk5_no_think_extractive_mixed_dev.json
  configs/experiments/1220_qwen3_8b_rag_e5_rerank1_no_think_extractive_mixed_dev.json
  configs/experiments/1221_qwen3_8b_rag_e5_rerank3_no_think_extractive_mixed_dev.json
  configs/experiments/1222_qwen3_8b_rag_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1223_qwen3_8b_3shot_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1224_qwen3_8b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1225_qwen3_8b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1226_qwen3_8b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1238_qwen3_8b_no_rag_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1239_qwen3_8b_rag_e5_topk1_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1240_qwen3_8b_rag_e5_topk3_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1241_qwen3_8b_rag_e5_topk5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1242_qwen3_8b_rag_e5_rerank1_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1243_qwen3_8b_rag_e5_rerank3_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1244_qwen3_8b_rag_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1245_qwen3_8b_3shot_no_rag_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1246_qwen3_8b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1247_qwen3_8b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1248_qwen3_8b_rag_3shot_e5_rerank5_no_think_extractive_mixed_sf_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job q3-8b-notk-gen finished at $(date)"
