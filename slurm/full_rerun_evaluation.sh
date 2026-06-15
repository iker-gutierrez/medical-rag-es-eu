#!/bin/bash
#SBATCH --job-name=medrag-eval-full
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_full_rerun_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_full_rerun_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

EVAL="python scripts/evaluate_predictions.py \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es"

echo "Evaluation started on $(hostname) at $(date)"

$EVAL \
  --predictions experiments/runs/17_mistral7b_no_rag_no_think_extractive_sf_dev/predictions.jsonl \
  --output reports/metrics/17_mistral7b_no_rag_no_think_extractive_sf_dev.json

$EVAL \
  --predictions experiments/runs/20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev/predictions.jsonl \
  --output reports/metrics/20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev.json

$EVAL \
  --predictions experiments/runs/28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev/predictions.jsonl \
  --output reports/metrics/28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev.json

$EVAL \
  --predictions experiments/runs/29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev/predictions.jsonl \
  --output reports/metrics/29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev.json

$EVAL \
  --predictions experiments/runs/30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev/predictions.jsonl \
  --output reports/metrics/30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev.json

$EVAL \
  --predictions experiments/runs/31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev/predictions.jsonl \
  --output reports/metrics/31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev.json

$EVAL \
  --predictions experiments/runs/32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev/predictions.jsonl \
  --output reports/metrics/32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev.json

$EVAL \
  --predictions experiments/runs/33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev/predictions.jsonl \
  --output reports/metrics/33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev.json

$EVAL \
  --predictions experiments/runs/36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev/predictions.jsonl \
  --output reports/metrics/36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev.json

$EVAL \
  --predictions experiments/runs/37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev/predictions.jsonl \
  --output reports/metrics/37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev.json

$EVAL \
  --predictions experiments/runs/38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev/predictions.jsonl \
  --output reports/metrics/38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev.json

echo "Evaluation finished at $(date)"
