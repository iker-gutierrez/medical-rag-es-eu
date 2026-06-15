#!/bin/bash
#SBATCH --job-name=medrag-eval-casi-mixed
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_casi_mixed_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_casi_mixed_%j.err
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
  --predictions experiments/runs/39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/40_mistral7b_rag_no_think_e5_topk1_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/40_mistral7b_rag_no_think_e5_topk1_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/42_mistral7b_rag_no_think_e5_topk5_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/42_mistral7b_rag_no_think_e5_topk5_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/43_mistral7b_rag_no_think_e5_rerank1_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/43_mistral7b_rag_no_think_e5_rerank1_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/44_mistral7b_rag_no_think_e5_rerank3_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/44_mistral7b_rag_no_think_e5_rerank3_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/45_mistral7b_rag_no_think_e5_rerank5_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/45_mistral7b_rag_no_think_e5_rerank5_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/46_mistral7b_random_3shot_no_rag_no_think_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/46_mistral7b_random_3shot_no_rag_no_think_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/47_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/47_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/48_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/48_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/49_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_casimedicos_dev/predictions.jsonl \
  --output reports/metrics/49_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_casimedicos_dev.json

$EVAL \
  --predictions experiments/runs/50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/51_mistral7b_rag_no_think_e5_topk1_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/51_mistral7b_rag_no_think_e5_topk1_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/52_mistral7b_rag_no_think_e5_topk3_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/52_mistral7b_rag_no_think_e5_topk3_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/53_mistral7b_rag_no_think_e5_topk5_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/53_mistral7b_rag_no_think_e5_topk5_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/54_mistral7b_rag_no_think_e5_rerank1_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/54_mistral7b_rag_no_think_e5_rerank1_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/55_mistral7b_rag_no_think_e5_rerank3_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/55_mistral7b_rag_no_think_e5_rerank3_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/56_mistral7b_rag_no_think_e5_rerank5_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/56_mistral7b_rag_no_think_e5_rerank5_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/57_mistral7b_random_3shot_no_rag_no_think_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/57_mistral7b_random_3shot_no_rag_no_think_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/59_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/59_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev.json

$EVAL \
  --predictions experiments/runs/60_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev/predictions.jsonl \
  --output reports/metrics/60_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev.json

echo "All evaluations done at $(date)"

# Update summary
python scripts/update_ablation_summary.py

echo "Summary updated at $(date)"
