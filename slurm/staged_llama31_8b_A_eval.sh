#!/bin/bash
#SBATCH --job-name=staged-eval-llama31_8b_A
#SBATCH --array=0-20%1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/staged_llama31_8b_A_eval_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/staged_llama31_8b_A_eval_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus
set -euo pipefail
mapfile -t TASKS < /home/igutierrez134/med_rag_thesis/experiments/staged_llama31_8b_A_eval_tasks.txt
ENTRY="${TASKS[$SLURM_ARRAY_TASK_ID]}"
PRED="$(echo "$ENTRY" | cut -d' ' -f2)"
OUT="$(echo "$ENTRY" | cut -d' ' -f3)"
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
scripts/pick_free_gpu.sh 2000 python scripts/evaluate_predictions_by_source.py \
  --predictions "$PRED" --output "$OUT" \
  --semantic-model "" \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang eu \
  --bertscore-device cuda:0
