#!/bin/bash
#SBATCH --job-name=medrag-casi-eu-retry
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/casimedicos_eu_retry_%A_task%a_%x.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/casimedicos_eu_retry_%A_task%a_%x.err
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

# Retry only the 6 failed tasks from gen_chain_casimedicos_eu (orig tasks 3,6,7,8,9,10)
CONFIGS=(
  "configs/experiments/75_latxa7b_rag_e5_topk5_extractive_sf_casimedicos_eu_dev.json"
  "configs/experiments/78_latxa7b_rag_e5_rerank5_extractive_sf_casimedicos_eu_dev.json"
  "configs/experiments/79_latxa7b_3shot_no_rag_extractive_sf_casimedicos_eu_dev.json"
  "configs/experiments/80_latxa7b_rag_3shot_e5_rerank5_extractive_sf_casimedicos_eu_dev.json"
  "configs/experiments/81_latxa7b_rag_sns1064_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev.json"
  "configs/experiments/82_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job finished at $(date)"
