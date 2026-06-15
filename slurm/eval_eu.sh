#!/bin/bash
#SBATCH --job-name=medrag-eval-eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_eu_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_eu_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# Use multilingual BERTScore; bert-base-multilingual-cased supports Basque
EVAL="python scripts/evaluate_predictions.py \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang eu"

echo "Basque evaluation started on $(hostname) at $(date)"

# --- SNS1064 EU (exps 61-71) ---
$EVAL \
  --predictions experiments/runs/61_latxa7b_no_rag_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/61_latxa7b_no_rag_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/62_latxa7b_rag_e5_topk1_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/62_latxa7b_rag_e5_topk1_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/63_latxa7b_rag_e5_topk3_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/63_latxa7b_rag_e5_topk3_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/64_latxa7b_rag_e5_topk5_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/64_latxa7b_rag_e5_topk5_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/65_latxa7b_rag_e5_rerank1_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/65_latxa7b_rag_e5_rerank1_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/66_latxa7b_rag_e5_rerank3_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/66_latxa7b_rag_e5_rerank3_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/67_latxa7b_rag_e5_rerank5_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/67_latxa7b_rag_e5_rerank5_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/68_latxa7b_3shot_no_rag_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/68_latxa7b_3shot_no_rag_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/69_latxa7b_rag_3shot_e5_rerank5_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/69_latxa7b_rag_3shot_e5_rerank5_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/70_latxa7b_rag_casimedicos_eu_e5_rerank5_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/70_latxa7b_rag_casimedicos_eu_e5_rerank5_extractive_sf_sns1064_eu_dev.json

$EVAL \
  --predictions experiments/runs/71_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_sns1064_eu_dev/predictions.jsonl \
  --output reports/metrics/71_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_sns1064_eu_dev.json

# --- CasiMedicos EU (exps 72-82) ---
$EVAL \
  --predictions experiments/runs/72_latxa7b_no_rag_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/72_latxa7b_no_rag_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/73_latxa7b_rag_e5_topk1_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/73_latxa7b_rag_e5_topk1_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/74_latxa7b_rag_e5_topk3_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/74_latxa7b_rag_e5_topk3_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/75_latxa7b_rag_e5_topk5_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/75_latxa7b_rag_e5_topk5_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/76_latxa7b_rag_e5_rerank1_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/76_latxa7b_rag_e5_rerank1_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/77_latxa7b_rag_e5_rerank3_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/77_latxa7b_rag_e5_rerank3_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/78_latxa7b_rag_e5_rerank5_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/78_latxa7b_rag_e5_rerank5_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/79_latxa7b_3shot_no_rag_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/79_latxa7b_3shot_no_rag_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/80_latxa7b_rag_3shot_e5_rerank5_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/80_latxa7b_rag_3shot_e5_rerank5_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/81_latxa7b_rag_sns1064_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/81_latxa7b_rag_sns1064_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev.json

$EVAL \
  --predictions experiments/runs/82_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev/predictions.jsonl \
  --output reports/metrics/82_latxa7b_rag_mixed_eu_e5_rerank5_extractive_sf_casimedicos_eu_dev.json

# --- Mixed EU (exps 83-93) ---
$EVAL \
  --predictions experiments/runs/83_latxa7b_no_rag_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/83_latxa7b_no_rag_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/84_latxa7b_rag_e5_topk1_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/84_latxa7b_rag_e5_topk1_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/85_latxa7b_rag_e5_topk3_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/85_latxa7b_rag_e5_topk3_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/86_latxa7b_rag_e5_topk5_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/86_latxa7b_rag_e5_topk5_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/87_latxa7b_rag_e5_rerank1_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/87_latxa7b_rag_e5_rerank1_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/88_latxa7b_rag_e5_rerank3_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/88_latxa7b_rag_e5_rerank3_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/89_latxa7b_rag_e5_rerank5_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/89_latxa7b_rag_e5_rerank5_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/90_latxa7b_3shot_no_rag_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/90_latxa7b_3shot_no_rag_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/91_latxa7b_rag_3shot_e5_rerank5_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/91_latxa7b_rag_3shot_e5_rerank5_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/92_latxa7b_rag_sns1064_eu_e5_rerank5_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/92_latxa7b_rag_sns1064_eu_e5_rerank5_extractive_sf_mixed_eu_dev.json

$EVAL \
  --predictions experiments/runs/93_latxa7b_rag_casimedicos_eu_e5_rerank5_extractive_sf_mixed_eu_dev/predictions.jsonl \
  --output reports/metrics/93_latxa7b_rag_casimedicos_eu_e5_rerank5_extractive_sf_mixed_eu_dev.json

echo "All Basque evaluations done at $(date)"

python scripts/update_eu_summary.py

echo "EU summary updated at $(date)"
