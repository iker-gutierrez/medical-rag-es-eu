#!/bin/bash
#SBATCH --job-name=refix-single-rec
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/refix_single_rec_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/refix_single_rec_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "refix-single-rec started on $(hostname) at $(date)"
nvidia-smi || true

python scripts/refix_single_feedback_record.py

echo ""
echo "=== Re-evaluating the affected run ==="
RUN_DIR="1279_qwen35_9b_rag_casimedicos_e5_rerank5_think_extractive_mixed_dev_seed42"
rm -f "reports/metrics/${RUN_DIR}.json" "reports/metrics/${RUN_DIR}_sns1064.json" "reports/metrics/${RUN_DIR}_casimedicos.json"

python scripts/evaluate_predictions.py \
  --predictions "experiments/runs/${RUN_DIR}/predictions.jsonl" \
  --output "reports/metrics/${RUN_DIR}.json" \
  --semantic-model intfloat/multilingual-e5-large \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es

echo ""
echo "=== Regenerating ES ablation report ==="
python scripts/write_mixed_es_seed_summary.py

echo ""
echo "refix-single-rec finished at $(date)"
