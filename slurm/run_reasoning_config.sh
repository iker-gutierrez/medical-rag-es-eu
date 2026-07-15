#!/bin/bash
#SBATCH --job-name=reasoning
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reasoning_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reasoning_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/run_reasoning_config.sh <config.json> [--seed N] [--limit N]" >&2
  exit 2
fi

CONFIG="$1"
shift

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Reasoning pipeline started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CONFIG=${CONFIG}"
echo "ARGS=$*"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

python scripts/run_reasoning_pipeline.py --config "$CONFIG" "$@"

echo "Reasoning pipeline finished at $(date)"
