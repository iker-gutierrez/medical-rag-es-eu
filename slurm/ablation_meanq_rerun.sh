#!/bin/bash
#SBATCH --job-name=abl-meanq
#SBATCH --array=0-26%1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_meanq_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_meanq_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
set -euo pipefail
mapfile -t TASKS < /home/igutierrez134/med_rag_thesis/experiments/meanq_rerun_tasks.txt
ENTRY="${TASKS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="${ENTRY%% *}"; SEED="${ENTRY##* }"
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
  --config "configs/experiments/${CONFIG}.json" --seed "${SEED}"
