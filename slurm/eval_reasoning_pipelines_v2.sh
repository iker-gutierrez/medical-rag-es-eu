#!/bin/bash
#SBATCH --job-name=eval-reasoning-v2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_reasoning_v2_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_reasoning_v2_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Evaluates the two config batches from tonight's reasoning-pipeline base-config
# corrections: ES switched to Qwen3.5-9B no-think + rerank5 (1330-1333), EU
# switched to Latxa + e5 top-1 (1320-1323). Supersedes the older
# eval_reasoning_pipelines.sh, which still points at the pre-correction configs
# (1300-1303 think mode, 1310-1313 Llama top-3).
#
# --semantic-model "" (cosine dropped): matches the current evaluation convention
# used everywhere else tonight, not the older template's multilingual-e5-large.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Reasoning-pipeline v2 evaluation started on $(hostname) at $(date)"

ES_RUNS=(
  1330_qwen35_9b_structured_cot_e5_rerank5_no_think_extractive_mixed_dev
  1331_qwen35_9b_thought_rag_e5_rerank5_no_think_extractive_mixed_dev
  1332_qwen35_9b_thought_rag_iter_e5_rerank5_no_think_extractive_mixed_dev
  1333_qwen35_9b_marag_e5_rerank5_no_think_extractive_mixed_dev
)
EU_RUNS=(
  1320_latxa_llama31_8b_structured_cot_e5_topk1_extractive_mixed_eu_dev
  1321_latxa_llama31_8b_thought_rag_e5_topk1_extractive_mixed_eu_dev
  1322_latxa_llama31_8b_thought_rag_iter_e5_topk1_extractive_mixed_eu_dev
  1323_latxa_llama31_8b_marag_e5_topk1_extractive_mixed_eu_dev
)

evaluate () {
  local run_dir="$1" lang="$2"
  local predictions="experiments/runs/${run_dir}/predictions.jsonl"
  local out="reports/metrics/${run_dir}.json"
  if [[ ! -f "$predictions" ]]; then echo "SKIP (no predictions): ${run_dir}" >&2; return 0; fi
  if [[ -f "$out" ]]; then echo "SKIP (already evaluated): ${run_dir}"; return 0; fi
  echo "Evaluating ${run_dir} (lang=${lang})"
  python scripts/evaluate_predictions.py \
    --predictions "$predictions" \
    --output "$out" \
    --semantic-model "" \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang "$lang"
}

for base in "${ES_RUNS[@]}"; do
  for seed in 42 43 44; do evaluate "${base}_seed${seed}" es; done
done
for base in "${EU_RUNS[@]}"; do
  for seed in 42 43 44; do evaluate "${base}_seed${seed}" eu; done
done

echo "Reasoning-pipeline v2 evaluation finished at $(date)"
