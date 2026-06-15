#!/bin/bash
#SBATCH --job-name=medrag-sf-fix12
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev12_promptfix_self_feedback_diagnostic_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/dev12_promptfix_self_feedback_diagnostic_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

RUN_ID="35_mistral7b_rag_no_think_e5_rerank5_extractive_promptfix_sf_dev12"
CONFIG="configs/experiments/${RUN_ID}.json"

echo "Limited prompt-fix self-feedback diagnostic started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

python scripts/run_generation_from_config.py --config "${CONFIG}"

python scripts/evaluate_predictions.py \
  --predictions "experiments/runs/${RUN_ID}/predictions.jsonl" \
  --output "reports/metrics/${RUN_ID}.json" \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es

python scripts/write_self_feedback_diagnostic.py \
  --metrics "reports/metrics/${RUN_ID}.json" \
  --predictions "experiments/runs/${RUN_ID}/predictions.jsonl" \
  --output "reports/metrics/self_feedback_promptfix_dev12_diagnostic.md"

echo "Limited prompt-fix self-feedback diagnostic finished at $(date)"
