#!/bin/bash
#SBATCH --job-name=eval-eu-mixed
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_mixed_eu_all_seeds_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_mixed_eu_all_seeds_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "EU mixed evaluation (all seeds) started on $(hostname) at $(date)"
nvidia-smi || true

# All Llama mixed EU dev runs (1040-1050) + all Latxa mixed EU dev runs (1051-1061)
# Includes seed42 (base), seed43, seed44 variants
RUN_DIRS=(
  1040_llama31_8b_no_rag_extractive_mixed_eu_dev
  1040_llama31_8b_no_rag_extractive_mixed_eu_dev_seed43
  1040_llama31_8b_no_rag_extractive_mixed_eu_dev_seed44
  1041_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev
  1041_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev_seed43
  1041_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev_seed44
  1042_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev
  1042_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev_seed43
  1042_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev_seed44
  1043_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev
  1043_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev_seed43
  1043_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev_seed44
  1044_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev
  1044_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev_seed43
  1044_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev_seed44
  1045_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev
  1045_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev_seed43
  1045_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev_seed44
  1046_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev
  1046_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev_seed43
  1046_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev_seed44
  1047_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev
  1047_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev_seed43
  1047_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev_seed44
  1048_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev
  1048_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev_seed43
  1048_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev_seed44
  1049_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev
  1049_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev_seed43
  1049_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev_seed44
  1050_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev
  1050_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev_seed43
  1050_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev_seed44
  1051_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev
  1051_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev_seed43
  1051_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev_seed44
  1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev
  1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev_seed43
  1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev_seed44
  1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev
  1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev_seed43
  1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev_seed44
  1054_latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev
  1054_latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev_seed43
  1054_latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev_seed44
  1055_latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev
  1055_latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev_seed43
  1055_latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev_seed44
  1056_latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev
  1056_latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev_seed43
  1056_latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev_seed44
  1057_latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev
  1057_latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev_seed43
  1057_latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev_seed44
  1058_latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev
  1058_latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev_seed43
  1058_latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev_seed44
  1059_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev
  1059_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev_seed43
  1059_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev_seed44
  1060_latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev
  1060_latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev_seed43
  1060_latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev_seed44
  1061_latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev
  1061_latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev_seed43
  1061_latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev_seed44
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
    --bertscore-lang eu
done

echo "Running report aggregation..."
python scripts/write_mixed_eu_seed_summary.py

echo "EU mixed evaluation finished at $(date)"
