#!/bin/bash
#SBATCH --job-name=eval-reasoning
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_reasoning_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_reasoning_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Reasoning-pipeline evaluation started on $(hostname) at $(date)"

ES_RUNS=(
  1300_qwen35_9b_structured_cot_e5_rerank5_extractive_mixed_dev
  1301_qwen35_9b_thought_rag_e5_rerank5_extractive_mixed_dev
  1302_qwen35_9b_thought_rag_iter_e5_rerank5_extractive_mixed_dev
  1303_qwen35_9b_marag_e5_rerank5_extractive_mixed_dev
)
EU_RUNS=(
  1310_llama31_8b_structured_cot_e5_topk3_extractive_mixed_eu_dev
  1311_llama31_8b_thought_rag_e5_topk3_extractive_mixed_eu_dev
  1312_llama31_8b_thought_rag_iter_e5_topk3_extractive_mixed_eu_dev
  1313_llama31_8b_marag_e5_topk3_extractive_mixed_eu_dev
)

# BERTScore language must match the run's language, exactly as the ES/EU ablation
# evaluations did -- scoring Basque output with the Spanish setting would not be
# comparable with the baseline rows these tables are measured against.
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
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang "$lang"
}

for base in "${ES_RUNS[@]}"; do
  for seed in 42 43 44; do evaluate "${base}_seed${seed}" es; done
done
for base in "${EU_RUNS[@]}"; do
  for seed in 42 43 44; do evaluate "${base}_seed${seed}" eu; done
done

echo "Writing summary..."
python scripts/write_reasoning_pipeline_summary.py

echo "Reasoning-pipeline evaluation finished at $(date)"
