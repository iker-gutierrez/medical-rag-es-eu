#!/bin/bash
#SBATCH --job-name=reeval-think-final
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reeval_think_final_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reeval_think_final_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "reeval-think-final started on $(hostname) at $(date)"

echo ""
echo "=== Deleting stale think-mode metric JSONs ==="
rm -f reports/metrics/127{0,1,2,3,4,5,6,7,8,9}_qwen35_9b*think*seed*.json
rm -f reports/metrics/1280_qwen35_9b*think*seed*.json

echo ""
echo "=== Re-evaluating think-mode predictions (corrected feedback pass) ==="
RUN_DIRS=(
  1270_qwen35_9b_no_rag_think_extractive_mixed_dev_seed42
  1270_qwen35_9b_no_rag_think_extractive_mixed_dev_seed43
  1270_qwen35_9b_no_rag_think_extractive_mixed_dev_seed44
  1271_qwen35_9b_rag_e5_topk1_think_extractive_mixed_dev_seed42
  1271_qwen35_9b_rag_e5_topk1_think_extractive_mixed_dev_seed43
  1271_qwen35_9b_rag_e5_topk1_think_extractive_mixed_dev_seed44
  1272_qwen35_9b_rag_e5_topk3_think_extractive_mixed_dev_seed42
  1272_qwen35_9b_rag_e5_topk3_think_extractive_mixed_dev_seed43
  1272_qwen35_9b_rag_e5_topk3_think_extractive_mixed_dev_seed44
  1273_qwen35_9b_rag_e5_topk5_think_extractive_mixed_dev_seed42
  1273_qwen35_9b_rag_e5_topk5_think_extractive_mixed_dev_seed43
  1273_qwen35_9b_rag_e5_topk5_think_extractive_mixed_dev_seed44
  1274_qwen35_9b_rag_e5_rerank1_think_extractive_mixed_dev_seed42
  1274_qwen35_9b_rag_e5_rerank1_think_extractive_mixed_dev_seed43
  1274_qwen35_9b_rag_e5_rerank1_think_extractive_mixed_dev_seed44
  1275_qwen35_9b_rag_e5_rerank3_think_extractive_mixed_dev_seed42
  1275_qwen35_9b_rag_e5_rerank3_think_extractive_mixed_dev_seed43
  1275_qwen35_9b_rag_e5_rerank3_think_extractive_mixed_dev_seed44
  1276_qwen35_9b_rag_e5_rerank5_think_extractive_mixed_dev_seed42
  1276_qwen35_9b_rag_e5_rerank5_think_extractive_mixed_dev_seed43
  1276_qwen35_9b_rag_e5_rerank5_think_extractive_mixed_dev_seed44
  1277_qwen35_9b_3shot_no_rag_think_extractive_mixed_dev_seed42
  1277_qwen35_9b_3shot_no_rag_think_extractive_mixed_dev_seed43
  1277_qwen35_9b_3shot_no_rag_think_extractive_mixed_dev_seed44
  1278_qwen35_9b_rag_sns1064_e5_rerank5_think_extractive_mixed_dev_seed42
  1278_qwen35_9b_rag_sns1064_e5_rerank5_think_extractive_mixed_dev_seed43
  1278_qwen35_9b_rag_sns1064_e5_rerank5_think_extractive_mixed_dev_seed44
  1279_qwen35_9b_rag_casimedicos_e5_rerank5_think_extractive_mixed_dev_seed42
  1279_qwen35_9b_rag_casimedicos_e5_rerank5_think_extractive_mixed_dev_seed43
  1279_qwen35_9b_rag_casimedicos_e5_rerank5_think_extractive_mixed_dev_seed44
  1280_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev_seed42
  1280_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev_seed43
  1280_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_mixed_dev_seed44
)

for run_dir in "${RUN_DIRS[@]}"; do
  predictions="experiments/runs/${run_dir}/predictions.jsonl"
  if [[ ! -f "$predictions" ]]; then
    echo "SKIP (no predictions): ${run_dir}" >&2
    continue
  fi
  out="reports/metrics/${run_dir}.json"
  if [[ -f "$out" ]]; then
    echo "SKIP (already evaluated): ${run_dir}"
    continue
  fi
  echo "Evaluating ${run_dir}"
  python scripts/evaluate_predictions.py \
    --predictions "$predictions" \
    --output "$out" \
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang es
done

echo ""
echo "=== Regenerating ES ablation report ==="
python scripts/write_mixed_es_seed_summary.py

echo ""
echo "reeval-think-final finished at $(date)"
