#!/bin/bash
#SBATCH --job-name=medrag-agentic-dev
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/agentic_dev_%A_task%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/agentic_dev_%A_task%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TASK_NAME="sns1064"
    RUN_ID="61_mistral7b_agentic_baseline_vs_exp8_sns1064_dev"
    BASELINE="experiments/runs/17_mistral7b_no_rag_no_think_extractive_sf_dev/predictions.jsonl"
    CANDIDATE="experiments/runs/38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev/predictions.jsonl"
    ;;
  1)
    TASK_NAME="casimedicos"
    RUN_ID="62_mistral7b_agentic_baseline_vs_exp2_casimedicos_dev"
    BASELINE="experiments/runs/39_mistral7b_no_rag_no_think_extractive_sf_casimedicos_dev/predictions.jsonl"
    CANDIDATE="experiments/runs/41_mistral7b_rag_no_think_e5_topk3_extractive_sf_casimedicos_dev/predictions.jsonl"
    ;;
  2)
    TASK_NAME="mixed"
    RUN_ID="63_mistral7b_agentic_baseline_vs_exp8_mixed_dev"
    BASELINE="experiments/runs/50_mistral7b_no_rag_no_think_extractive_sf_mixed_dev/predictions.jsonl"
    CANDIDATE="experiments/runs/58_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_mixed_dev/predictions.jsonl"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "Agentic dev task started on $(hostname) at $(date)"
echo "TASK_NAME=${TASK_NAME}"
echo "RUN_ID=${RUN_ID}"
echo "BASELINE=${BASELINE}"
echo "CANDIDATE=${CANDIDATE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

python scripts/run_agentic_reasoner.py \
  --experiment-name "${RUN_ID}" \
  --baseline-predictions "${BASELINE}" \
  --candidate-predictions "${CANDIDATE}" \
  --output "experiments/runs/${RUN_ID}/predictions.jsonl" \
  --source-answer-key initial \
  --max-new-tokens 512

python scripts/evaluate_predictions.py \
  --predictions "experiments/runs/${RUN_ID}/predictions.jsonl" \
  --output "reports/metrics/${RUN_ID}.json" \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es

echo "Agentic dev task finished at $(date)"
