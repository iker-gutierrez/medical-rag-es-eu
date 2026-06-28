#!/bin/bash
#SBATCH --job-name=llama-latxa-eu-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llama_latxa_eu_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llama_latxa_eu_evaluation_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
shopt -s nullglob

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Llama/Latxa EU evaluation started on $(hostname) at $(date)"

for run_number in $(seq 106 171); do
  matches=(experiments/runs/${run_number}_*/predictions.jsonl)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "Expected exactly one predictions file for run number ${run_number}, found ${#matches[@]}" >&2
    exit 2
  fi
  predictions="${matches[0]}"
  run_id="$(basename "$(dirname "$predictions")")"
  echo "Evaluating ${run_id}"
  python scripts/evaluate_predictions.py \
    --predictions "$predictions" \
    --output "reports/metrics/${run_id}.json" \
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang eu
done

python scripts/write_eu_ablation_summary.py

echo "Llama/Latxa EU evaluation finished at $(date)"
