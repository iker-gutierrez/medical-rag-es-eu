#!/bin/bash
#SBATCH --job-name=eval-abl
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:2
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_ablation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_ablation_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Evaluation does NOT measure latency (it only computes metric scores from stored
# predictions), so unlike generation it may use both GPUs. The 165 evaluations are
# sharded across the two cards and run in parallel; each evaluation loads BERTScore
# + multilingual-e5 on its assigned GPU. Generation latency is untouched: this reads
# predictions that were already produced one-GPU-per-task.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Seeded-ablation evaluation started on $(hostname) at $(date)"

# Gate: a truncated generation still gets scored -- on a half-written answer. Catch
# it here rather than let it become a quietly depressed row in a results table.
echo ""
echo "=== truncation gate ==="
python scripts/check_no_truncation.py

CONFIGS=$(cat /tmp/claude-1034/-home-igutierrez134/033c132e-359b-4152-9f88-238f66c6f423/scratchpad/new_configs.txt)

# Evaluate one shard of (config, seed) pairs, all on a single assigned GPU.
# shard_id in {0,1}; a pair is handled by this shard when (line index %% 2)==shard_id,
# which balances the two GPUs regardless of how ES/EU configs are ordered.
evaluate_shard () {
  local gpu="$1" shard="$2" idx=0
  for cfg in $CONFIGS; do
    if [[ "$cfg" == *_eu_dev ]]; then local lang=eu; else local lang=es; fi
    for seed in 42 43 44; do
      if (( idx %% 2 == shard )); then
        local run="${cfg}_seed${seed}"
        local predictions="experiments/runs/${run}/predictions.jsonl"
        local out="reports/metrics/${run}.json"
        if [[ -f "$predictions" && ! -f "$out" ]]; then
          echo "[GPU ${gpu}] evaluating ${run} (lang=${lang})"
          CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_predictions.py \
            --predictions "$predictions" \
            --output "$out" \
            --semantic-model "" \
            --bertscore-model bert-base-multilingual-cased \
            --bertscore-lang "$lang"
        fi
      fi
      idx=$((idx + 1))
    done
  done
}

echo ""
echo "=== evaluating 165 runs across 2 GPUs in parallel ==="
evaluate_shard 0 0 &
PID0=$!
evaluate_shard 1 1 &
PID1=$!
wait $PID0
wait $PID1

echo ""
echo "=== regenerating decision tables (same format, new seeded values) ==="
python scripts/write_mixed_es_seed_summary.py
python scripts/write_mixed_eu_seed_summary.py

echo ""
echo "Seeded-ablation evaluation finished at $(date)"
