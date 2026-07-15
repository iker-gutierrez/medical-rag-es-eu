#!/bin/bash
#SBATCH --job-name=eval-reason
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=96GB
#SBATCH --gres=gpu:2
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_reason_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_reason_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Evaluate the seeded reasoning-pipeline reruns. Evaluation does not measure latency,
# so it uses both GPUs (the two-GPU shard mirrors eval_ablation_rerun.sh). The
# generation latency the pipelines report was fixed at one-GPU-per-task and is not
# touched here.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Reasoning-pipeline evaluation started on $(hostname) at $(date)"

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

# Each shard evaluates one language on one GPU, in parallel. BERTScore language must
# match the run's language (es / eu), as everywhere else in this thesis.
evaluate_lang () {
  local gpu="$1" lang="$2"; shift 2
  local configs=("$@")
  for cfg in "${configs[@]}"; do
    for seed in 42 43 44; do
      local run="${cfg}_seed${seed}"
      local predictions="experiments/runs/${run}/predictions.jsonl"
      local out="reports/metrics/${run}.json"
      if [[ -f "$predictions" && ! -f "$out" ]]; then
        echo "[GPU ${gpu}] evaluating ${run}"
        CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_predictions.py \
          --predictions "$predictions" \
          --output "$out" \
          --semantic-model "" \
          --bertscore-model bert-base-multilingual-cased \
          --bertscore-lang "$lang"
      fi
    done
  done
}

echo ""
echo "=== evaluating ES (GPU 0) and EU (GPU 1) in parallel ==="
evaluate_lang 0 es "${ES_CONFIGS[@]}" &
PID0=$!
evaluate_lang 1 eu "${EU_CONFIGS[@]}" &
PID1=$!
wait $PID0
wait $PID1

echo ""
echo "=== regenerating reasoning-pipeline tables ==="
python scripts/write_reasoning_latex_table.py
python scripts/write_reasoning_pipeline_summary.py || true

echo ""
echo "Reasoning-pipeline evaluation finished at $(date)"
