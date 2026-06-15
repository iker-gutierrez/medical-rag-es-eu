#!/bin/bash
#SBATCH --job-name=medrag-rerank-v2-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev_reranker_topk_v2_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev_reranker_topk_v2_evaluation_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Reranker/top-k v2 evaluation started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

for run_id in \
  32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev \
  30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev \
  33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev \
  28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev \
  31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev \
  29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev
do
  python scripts/evaluate_predictions.py \
    --predictions "experiments/runs/${run_id}/predictions.jsonl" \
    --output "reports/metrics/${run_id}.json" \
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang es
done

python scripts/write_supervisor_summary.py \
  --csv-output reports/metrics/dev_ablation_results.csv \
  --markdown-output reports/metrics/dev_ablation_results.md

echo "Reranker/top-k v2 evaluation finished at $(date)"
