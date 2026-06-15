#!/bin/bash
#SBATCH --job-name=medrag-exp67-retry
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/exp67_retry_%j_%x.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/exp67_retry_%j_%x.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

# exp 67: rerank5 SNS1064 EU — OOMed before truncation fix in generation.py
echo "Running config: configs/experiments/67_latxa7b_rag_e5_rerank5_extractive_sf_sns1064_eu_dev.json"
python scripts/run_generation_from_config.py \
  --config configs/experiments/67_latxa7b_rag_e5_rerank5_extractive_sf_sns1064_eu_dev.json

echo "Job finished at $(date)"
