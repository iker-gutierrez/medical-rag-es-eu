#!/bin/bash
#SBATCH --job-name=eval-es-mixed
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_mixed_es_all_seeds_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_mixed_es_all_seeds_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "ES mixed evaluation (all seeds) started on $(hostname) at $(date)"
nvidia-smi || true

# All Qwen3.5-9B mixed dev runs (1128-1138) + all Mistral mixed dev runs (1260-1270)
# Includes seed42 (base), seed43, seed44 variants
RUN_DIRS=(
  1128_qwen35_9b_no_rag_no_think_extractive_mixed_dev
  1128_qwen35_9b_no_rag_no_think_extractive_mixed_dev_seed43
  1128_qwen35_9b_no_rag_no_think_extractive_mixed_dev_seed44
  1129_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev
  1129_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev_seed43
  1129_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev_seed44
  1130_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev
  1130_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev_seed43
  1130_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev_seed44
  1131_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev
  1131_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev_seed43
  1131_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev_seed44
  1132_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev
  1132_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev_seed43
  1132_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev_seed44
  1133_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev
  1133_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev_seed43
  1133_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev_seed44
  1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev
  1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev_seed44
  1135_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev
  1135_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev_seed43
  1135_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev_seed44
  1136_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev
  1136_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1136_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev_seed44
  1137_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev
  1137_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1137_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev_seed44
  1138_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev
  1138_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1138_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev_seed44
  1260_mistral7b_no_rag_no_think_extractive_mixed_dev
  1260_mistral7b_no_rag_no_think_extractive_mixed_dev_seed43
  1260_mistral7b_no_rag_no_think_extractive_mixed_dev_seed44
  1261_mistral7b_rag_e5_topk1_no_think_extractive_mixed_dev
  1261_mistral7b_rag_e5_topk1_no_think_extractive_mixed_dev_seed43
  1261_mistral7b_rag_e5_topk1_no_think_extractive_mixed_dev_seed44
  1262_mistral7b_rag_e5_topk3_no_think_extractive_mixed_dev
  1262_mistral7b_rag_e5_topk3_no_think_extractive_mixed_dev_seed43
  1262_mistral7b_rag_e5_topk3_no_think_extractive_mixed_dev_seed44
  1263_mistral7b_rag_e5_topk5_no_think_extractive_mixed_dev
  1263_mistral7b_rag_e5_topk5_no_think_extractive_mixed_dev_seed43
  1263_mistral7b_rag_e5_topk5_no_think_extractive_mixed_dev_seed44
  1264_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_dev
  1264_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_dev_seed43
  1264_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_dev_seed44
  1265_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_dev
  1265_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_dev_seed43
  1265_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_dev_seed44
  1266_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_dev
  1266_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1266_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_dev_seed44
  1267_mistral7b_3shot_no_rag_no_think_extractive_mixed_dev
  1267_mistral7b_3shot_no_rag_no_think_extractive_mixed_dev_seed43
  1267_mistral7b_3shot_no_rag_no_think_extractive_mixed_dev_seed44
  1268_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev
  1268_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1268_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev_seed44
  1269_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev
  1269_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1269_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev_seed44
  1270_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev
  1270_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev_seed43
  1270_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev_seed44
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

echo "Running report aggregation..."
python scripts/write_mixed_es_seed_summary.py

echo "ES mixed evaluation finished at $(date)"
