#!/bin/bash
# Launch the full reasoning-pipeline dev grid: 4 pipelines x 2 languages x 3 seeds.
#
# Submitted as two chains (one per language) rather than 24 independent jobs, so
# the cluster only ever sees 2 of our jobs at a time and each chain reuses nothing
# it shouldn't. Usage:
#
#   bash slurm/run_reasoning_all.sh          # submit
#   bash slurm/run_reasoning_all.sh --dry    # print what would be submitted
set -euo pipefail
cd /home/igutierrez134/med_rag_thesis

DRY=""
[[ "${1:-}" == "--dry" ]] && DRY=1

ES_CONFIGS=(
  1300_qwen35_9b_structured_cot_e5_rerank5_extractive_mixed_dev
  1301_qwen35_9b_thought_rag_e5_rerank5_extractive_mixed_dev
  1302_qwen35_9b_thought_rag_iter_e5_rerank5_extractive_mixed_dev
  1303_qwen35_9b_marag_e5_rerank5_extractive_mixed_dev
)
EU_CONFIGS=(
  1310_llama31_8b_structured_cot_e5_topk3_extractive_mixed_eu_dev
  1311_llama31_8b_thought_rag_e5_topk3_extractive_mixed_eu_dev
  1312_llama31_8b_thought_rag_iter_e5_topk3_extractive_mixed_eu_dev
  1313_llama31_8b_marag_e5_topk3_extractive_mixed_eu_dev
)
SEEDS=(42 43 44)

submit_chain () {
  local -n configs=$1
  local chain_name=$2
  local dep=""
  for cfg in "${configs[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local out="experiments/runs/${cfg}_seed${seed}/predictions.jsonl"
      if [[ -f "$out" ]]; then
        echo "  SKIP (exists): ${cfg} seed${seed}"
        continue
      fi
      local args=(--job-name="rp-${chain_name}")
      [[ -n "$dep" ]] && args+=(--dependency=afterany:"$dep")
      if [[ -n "$DRY" ]]; then
        echo "  sbatch ${args[*]} slurm/run_reasoning_config.sh configs/experiments/${cfg}.json --seed ${seed}"
        dep="DRY"
        continue
      fi
      local jid
      jid=$(sbatch --parsable "${args[@]}" slurm/run_reasoning_config.sh \
              "configs/experiments/${cfg}.json" --seed "${seed}")
      echo "  submitted ${jid}: ${cfg} seed${seed}${dep:+ (after ${dep})}"
      dep="$jid"
    done
  done
  LAST_JOB="$dep"
}

echo "ES chain (Qwen3.5-9B, Spanish):"
submit_chain ES_CONFIGS es
ES_LAST="$LAST_JOB"

echo "EU chain (Llama-3.1-8B, Basque):"
submit_chain EU_CONFIGS eu
EU_LAST="$LAST_JOB"

if [[ -z "$DRY" && -n "$ES_LAST" && -n "$EU_LAST" ]]; then
  eval_jid=$(sbatch --parsable --dependency=afterany:"${ES_LAST}":"${EU_LAST}" \
    slurm/eval_reasoning_pipelines.sh)
  echo ""
  echo "submitted ${eval_jid}: evaluation + summary (after both chains)"
fi
