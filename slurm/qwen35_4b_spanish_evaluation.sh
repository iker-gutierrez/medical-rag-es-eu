#!/bin/bash
#SBATCH --job-name=qwen35-4b-es-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_4b_spanish_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_4b_spanish_evaluation_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

RUN_IDS=(
  94_qwen35_4b_no_rag_no_think_extractive_sns1064_dev
  95_qwen35_4b_no_rag_think_extractive_sns1064_dev
  96_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_sns1064_dev
  97_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_sns1064_dev
  98_qwen35_4b_no_rag_no_think_extractive_casimedicos_dev
  99_qwen35_4b_no_rag_think_extractive_casimedicos_dev
  100_qwen35_4b_rag_no_think_e5_topk3_extractive_casimedicos_dev
  101_qwen35_4b_rag_think_e5_topk3_extractive_casimedicos_dev
  102_qwen35_4b_no_rag_no_think_extractive_mixed_dev
  103_qwen35_4b_no_rag_think_extractive_mixed_dev
  104_qwen35_4b_rag_3shot_no_think_e5_rerank5_extractive_mixed_dev
  105_qwen35_4b_rag_3shot_think_e5_rerank5_extractive_mixed_dev
)

echo "Qwen3.5-4B Spanish evaluation started on $(hostname) at $(date)"

for run_id in "${RUN_IDS[@]}"; do
  echo "Evaluating ${run_id}"
  python scripts/evaluate_predictions.py \
    --predictions "experiments/runs/${run_id}/predictions.jsonl" \
    --output "reports/metrics/${run_id}.json" \
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang es
done

python scripts/write_qwen35_spanish_summary.py

echo "Qwen3.5-4B Spanish evaluation finished at $(date)"
