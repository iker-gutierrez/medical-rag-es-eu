#!/bin/bash
#SBATCH --job-name=mist-notk-gen
#SBATCH --array=0-21%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_no_think_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_no_think_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Job mist-notk-gen started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

CONFIGS=(
  configs/experiments/1260_mistral7b_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1261_mistral7b_rag_e5_topk1_no_think_extractive_mixed_dev.json
  configs/experiments/1262_mistral7b_rag_e5_topk3_no_think_extractive_mixed_dev.json
  configs/experiments/1263_mistral7b_rag_e5_topk5_no_think_extractive_mixed_dev.json
  configs/experiments/1264_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_dev.json
  configs/experiments/1265_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_dev.json
  configs/experiments/1266_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1267_mistral7b_3shot_no_rag_no_think_extractive_mixed_dev.json
  configs/experiments/1268_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1269_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1270_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev.json
  configs/experiments/1271_mistral7b_no_rag_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1272_mistral7b_rag_e5_topk1_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1273_mistral7b_rag_e5_topk3_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1274_mistral7b_rag_e5_topk5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1275_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1276_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1277_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1278_mistral7b_3shot_no_rag_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1279_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1280_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  configs/experiments/1281_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_sf_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CONFIG"

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Job mist-notk-gen finished at $(date)"
