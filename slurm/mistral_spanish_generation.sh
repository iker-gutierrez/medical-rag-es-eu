#!/bin/bash
#SBATCH --job-name=mistral-es-gen
#SBATCH --array=0-32%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_spanish_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_spanish_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

CONFIGS=(
  configs/experiments/17_mistral7b_no_rag_no_think_extractive_sf_dev.json
  configs/experiments/32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev.json
  configs/experiments/30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev.json
  configs/experiments/33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev.json
  configs/experiments/28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev.json
  configs/experiments/31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev.json
  configs/experiments/29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev.json
  configs/experiments/20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev.json
  configs/experiments/36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev.json
  configs/experiments/37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev.json
  configs/experiments/38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev.json
  configs/experiments/39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev.json
  configs/experiments/40_mistral7b_rag_no_think_e5_topk1_extractive_sf_casimedicos_dev.json
  configs/experiments/41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev.json
  configs/experiments/42_mistral7b_rag_no_think_e5_topk5_extractive_sf_casimedicos_dev.json
  configs/experiments/43_mistral7b_rag_no_think_e5_rerank1_extractive_sf_casimedicos_dev.json
  configs/experiments/44_mistral7b_rag_no_think_e5_rerank3_extractive_sf_casimedicos_dev.json
  configs/experiments/45_mistral7b_rag_no_think_e5_rerank5_extractive_sf_casimedicos_dev.json
  configs/experiments/46_mistral7b_random_3shot_no_rag_no_think_extractive_sf_casimedicos_dev.json
  configs/experiments/47_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_casimedicos_dev.json
  configs/experiments/48_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_casimedicos_dev.json
  configs/experiments/49_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_casimedicos_dev.json
  configs/experiments/50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev.json
  configs/experiments/51_mistral7b_rag_no_think_e5_topk1_extractive_sf_mixed_dev.json
  configs/experiments/52_mistral7b_rag_no_think_e5_topk3_extractive_sf_mixed_dev.json
  configs/experiments/53_mistral7b_rag_no_think_e5_topk5_extractive_sf_mixed_dev.json
  configs/experiments/54_mistral7b_rag_no_think_e5_rerank1_extractive_sf_mixed_dev.json
  configs/experiments/55_mistral7b_rag_no_think_e5_rerank3_extractive_sf_mixed_dev.json
  configs/experiments/56_mistral7b_rag_no_think_e5_rerank5_extractive_sf_mixed_dev.json
  configs/experiments/57_mistral7b_random_3shot_no_rag_no_think_extractive_sf_mixed_dev.json
  configs/experiments/58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev.json
  configs/experiments/59_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev.json
  configs/experiments/60_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Mistral Spanish generation started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Mistral Spanish generation finished at $(date)"
