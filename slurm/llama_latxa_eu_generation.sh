#!/bin/bash
#SBATCH --job-name=llama-latxa-eu-gen
#SBATCH --array=0-65%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llama_latxa_eu_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llama_latxa_eu_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
shopt -s nullglob

run_number=$((106 + SLURM_ARRAY_TASK_ID))
matches=(configs/experiments/${run_number}_*_dev.json)
if [[ ${#matches[@]} -ne 1 ]]; then
  echo "Expected exactly one config for run number ${run_number}, found ${#matches[@]}" >&2
  exit 2
fi
CONFIG="${matches[0]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Llama/Latxa EU generation started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

python scripts/run_generation_from_config.py --config "$CONFIG"

echo "Llama/Latxa EU generation finished at $(date)"
