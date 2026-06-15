#!/bin/bash
# Submit all 11 generation jobs as a sequential chain, then evaluation.
# Run from project root: bash slurm/submit_full_rerun.sh

set -euo pipefail

LOGDIR=experiments/slurm_logs

CONFIGS=(
  "configs/experiments/17_mistral7b_no_rag_no_think_extractive_sf_dev.json"
  "configs/experiments/20_mistral7b_random_3shot_no_rag_no_think_extractive_sf_dev.json"
  "configs/experiments/28_mistral7b_rag_no_think_e5_rerank1_extractive_v2_sf_dev.json"
  "configs/experiments/29_mistral7b_rag_no_think_e5_rerank5_extractive_v2_sf_dev.json"
  "configs/experiments/30_mistral7b_rag_no_think_e5_topk3_extractive_v2_sf_dev.json"
  "configs/experiments/31_mistral7b_rag_no_think_e5_rerank3_extractive_v2_sf_dev.json"
  "configs/experiments/32_mistral7b_rag_no_think_e5_topk1_extractive_v2_sf_dev.json"
  "configs/experiments/33_mistral7b_rag_no_think_e5_topk5_extractive_v2_sf_dev.json"
  "configs/experiments/36_mistral7b_rag_no_think_casimedicos_e5_rerank5_extractive_sf_dev.json"
  "configs/experiments/37_mistral7b_rag_no_think_sns1064_casimedicos_e5_rerank5_extractive_sf_dev.json"
  "configs/experiments/38_mistral7b_rag_random_3shot_no_think_e5_rerank5_extractive_sf_dev.json"
)

NAMES=(
  "17_no_rag_no_think"
  "20_random_3shot_no_rag"
  "28_e5_rerank1"
  "29_e5_rerank5"
  "30_e5_top3"
  "31_e5_rerank3"
  "32_e5_top1"
  "33_e5_top5"
  "36_casimedicos_e5_rerank5"
  "37_mixed_e5_rerank5"
  "38_rag_3shot_e5_rerank5"
)

PREV_JOB=""
ALL_JOB_IDS=()

for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  NAME="${NAMES[$i]}"

  SBATCH_ARGS=(
    --job-name="medrag-${NAME}"
    --output="${LOGDIR}/gen_${NAME}_%j.log"
    --error="${LOGDIR}/gen_${NAME}_%j.err"
  )

  if [[ -n "$PREV_JOB" ]]; then
    SBATCH_ARGS+=(--dependency="afterok:${PREV_JOB}")
  fi

  JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" slurm/run_generation_config.sh "$CONFIG" | awk '{print $NF}')
  echo "Submitted ${NAME} → job ${JOB_ID}"
  PREV_JOB="$JOB_ID"
  ALL_JOB_IDS+=("$JOB_ID")
done

DEP_STRING=$(printf ":%s" "${ALL_JOB_IDS[@]}")
DEP_STRING="afterok${DEP_STRING}"

EVAL_JOB=$(sbatch \
  --dependency="$DEP_STRING" \
  slurm/full_rerun_evaluation.sh | awk '{print $NF}')
echo "Submitted evaluation → job ${EVAL_JOB} (after all 11 generation jobs)"
echo ""
echo "All generation job IDs: ${ALL_JOB_IDS[*]}"
echo "Evaluation job ID: ${EVAL_JOB}"
