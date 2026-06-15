#!/bin/bash
#SBATCH --job-name=medrag-think-topk-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev_think_topk_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev_think_topk_evaluation_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Think top-k evaluation started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

for run_id in \
  26_mistral7b_rag_think_topk1_sf_dev \
  27_mistral7b_rag_think_topk5_sf_dev
do
  python scripts/evaluate_predictions.py \
    --predictions "experiments/runs/${run_id}/predictions.jsonl" \
    --output "reports/metrics/${run_id}.json" \
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang es
done

python scripts/summarize_metrics.py \
  --input \
    reports/metrics/04_mistral7b_no_rag_no_think_sf_dev.json \
    reports/metrics/05_mistral7b_rag_no_think_sf_dev.json \
    reports/metrics/06_mistral7b_no_rag_think_sf_dev.json \
    reports/metrics/07_mistral7b_rag_think_sf_dev.json \
    reports/metrics/12_mistral7b_rag_no_think_topk1_sf_dev.json \
    reports/metrics/13_mistral7b_rag_no_think_topk5_sf_dev.json \
    reports/metrics/14_mistral7b_rag_no_think_e5_topk3_sf_dev.json \
    reports/metrics/15_mistral7b_random_3shot_no_rag_no_think_sf_dev.json \
    reports/metrics/16_mistral7b_retrieval_3shot_no_rag_no_think_sf_dev.json \
    reports/metrics/17_mistral7b_no_rag_no_think_extractive_sf_dev.json \
    reports/metrics/18_mistral7b_rag_no_think_extractive_sf_dev.json \
    reports/metrics/19_mistral7b_rag_no_think_e5_topk3_extractive_sf_dev.json \
    reports/metrics/20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev.json \
    reports/metrics/21_mistral7b_retrieval_3shot_no_rag_no_think_extractive_sf_dev.json \
    reports/metrics/22_mistral7b_rag_no_think_e5_rerank3_v1_extractive_sf_dev.json \
    reports/metrics/23_mistral7b_rag_no_think_minilm_rerank3_v1_extractive_sf_dev.json \
    reports/metrics/24_mistral7b_rag_no_think_casimedicos_e5_rerank3_v1_extractive_sf_on_sns_dev.json \
    reports/metrics/25_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank3_v1_extractive_sf_on_sns_dev.json \
    reports/metrics/26_mistral7b_rag_think_topk1_sf_dev.json \
    reports/metrics/27_mistral7b_rag_think_topk5_sf_dev.json \
  --output reports/metrics/dev_ablation_results.csv \
  --markdown-output reports/metrics/dev_ablation_results.md

echo "Think top-k evaluation finished at $(date)"
