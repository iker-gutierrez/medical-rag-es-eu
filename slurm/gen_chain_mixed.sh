#!/bin/bash
#SBATCH --job-name=medrag-mixed-dev
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-10%1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mixed_dev_%A_task%a_%x.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mixed_dev_%A_task%a_%x.err
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
  "configs/experiments/50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev.json"
  "configs/experiments/51_mistral7b_rag_no_think_e5_topk1_extractive_sf_mixed_dev.json"
  "configs/experiments/52_mistral7b_rag_no_think_e5_topk3_extractive_sf_mixed_dev.json"
  "configs/experiments/53_mistral7b_rag_no_think_e5_topk5_extractive_sf_mixed_dev.json"
  "configs/experiments/54_mistral7b_rag_no_think_e5_rerank1_extractive_sf_mixed_dev.json"
  "configs/experiments/55_mistral7b_rag_no_think_e5_rerank3_extractive_sf_mixed_dev.json"
  "configs/experiments/56_mistral7b_rag_no_think_e5_rerank5_extractive_sf_mixed_dev.json"
  "configs/experiments/57_mistral7b_random_3shot_no_rag_no_think_extractive_sf_mixed_dev.json"
  "configs/experiments/58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev.json"
  "configs/experiments/59_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev.json"
  "configs/experiments/60_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev.json"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job finished at $(date)"
