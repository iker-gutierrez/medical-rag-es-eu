#!/bin/bash
#SBATCH --job-name=medrag-sns-eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-10%1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/sns1064_eu_%A_task%a_%x.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/sns1064_eu_%A_task%a_%x.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

CONFIGS=(
  "configs/experiments/61_latxa7b_no_rag_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/62_latxa7b_rag_e5_topk1_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/63_latxa7b_rag_e5_topk3_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/64_latxa7b_rag_e5_topk5_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/65_latxa7b_rag_e5_rerank1_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/66_latxa7b_rag_e5_rerank3_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/67_latxa7b_rag_e5_rerank5_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/68_latxa7b_3shot_no_rag_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/69_latxa7b_rag_3shot_e5_rerank5_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/70_latxa7b_rag_casimedicos_eu_e5_rerank5_extractive_sf_sns1064_eu_dev.json"
  "configs/experiments/71_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_sns1064_eu_dev.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job finished at $(date)"
