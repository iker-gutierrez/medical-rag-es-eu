#!/bin/bash
#SBATCH --job-name=mistral-es-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_spanish_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_spanish_evaluation_%j.err
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
  17_mistral7b_no_rag_no_think_extractive_sf_dev
  32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev
  30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev
  33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev
  28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev
  31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev
  29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev
  20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev
  36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev
  37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev
  38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev
  39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev
  40_mistral7b_rag_no_think_e5_topk1_extractive_sf_casimedicos_dev
  41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev
  42_mistral7b_rag_no_think_e5_topk5_extractive_sf_casimedicos_dev
  43_mistral7b_rag_no_think_e5_rerank1_extractive_sf_casimedicos_dev
  44_mistral7b_rag_no_think_e5_rerank3_extractive_sf_casimedicos_dev
  45_mistral7b_rag_no_think_e5_rerank5_extractive_sf_casimedicos_dev
  46_mistral7b_random_3shot_no_rag_no_think_extractive_sf_casimedicos_dev
  47_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_casimedicos_dev
  48_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_casimedicos_dev
  49_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_casimedicos_dev
  50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev
  51_mistral7b_rag_no_think_e5_topk1_extractive_sf_mixed_dev
  52_mistral7b_rag_no_think_e5_topk3_extractive_sf_mixed_dev
  53_mistral7b_rag_no_think_e5_topk5_extractive_sf_mixed_dev
  54_mistral7b_rag_no_think_e5_rerank1_extractive_sf_mixed_dev
  55_mistral7b_rag_no_think_e5_rerank3_extractive_sf_mixed_dev
  56_mistral7b_rag_no_think_e5_rerank5_extractive_sf_mixed_dev
  57_mistral7b_random_3shot_no_rag_no_think_extractive_sf_mixed_dev
  58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev
  59_mistral7b_rag_no_think_sns1064_e5_rerank5_extractive_sf_mixed_dev
  60_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_mixed_dev
)

echo "Mistral Spanish evaluation started on $(hostname) at $(date)"

for run_id in "${RUN_IDS[@]}"; do
  echo "Evaluating ${run_id}"
  python scripts/evaluate_predictions.py \
    --predictions "experiments/runs/${run_id}/predictions.jsonl" \
    --output "reports/metrics/${run_id}.json" \
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang es
done

python scripts/write_supervisor_summary.py
python scripts/update_ablation_summary.py

echo "Mistral Spanish evaluation and report update finished at $(date)"
