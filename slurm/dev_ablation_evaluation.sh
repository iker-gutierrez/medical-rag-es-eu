#!/bin/bash
#SBATCH --job-name=medrag-dev-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev_ablation_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev_ablation_evaluation_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Evaluation started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

python scripts/evaluate_predictions.py \
  --predictions experiments/runs/04_mistral7b_no_rag_no_think_sf_dev/predictions.jsonl \
  --output reports/metrics/04_mistral7b_no_rag_no_think_sf_dev.json \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es

python scripts/evaluate_predictions.py \
  --predictions experiments/runs/05_mistral7b_rag_no_think_sf_dev/predictions.jsonl \
  --output reports/metrics/05_mistral7b_rag_no_think_sf_dev.json \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es

python scripts/evaluate_predictions.py \
  --predictions experiments/runs/06_mistral7b_no_rag_think_sf_dev/predictions.jsonl \
  --output reports/metrics/06_mistral7b_no_rag_think_sf_dev.json \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es

python scripts/evaluate_predictions.py \
  --predictions experiments/runs/07_mistral7b_rag_think_sf_dev/predictions.jsonl \
  --output reports/metrics/07_mistral7b_rag_think_sf_dev.json \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es

echo "Evaluation finished at $(date)"
