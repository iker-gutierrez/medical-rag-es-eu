#!/bin/bash
#SBATCH --job-name=reason-smoke
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reasoning_smoke_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reasoning_smoke_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Smoke test started on $(hostname) at $(date)"
nvidia-smi || true

# 6 records is enough to exercise every path: the mixed dev set interleaves
# SNS1064 (open answer -> semantic conflict) and CasiMedicos (multiple choice ->
# option disagreement), so both conflict modes get hit.
# One EU pipeline per config, into throwaway _smoke run dirs.
for cfg in \
  1310_llama31_8b_structured_cot_e5_rerank5_extractive_mixed_eu_dev \
  1311_llama31_8b_thought_rag_e5_rerank5_extractive_mixed_eu_dev \
  1312_llama31_8b_thought_rag_iter_e5_rerank5_extractive_mixed_eu_dev \
  1313_llama31_8b_marag_e5_rerank5_extractive_mixed_eu_dev
do
  echo ""
  echo "######## SMOKE: ${cfg}"
  python scripts/run_reasoning_pipeline.py \
    --config "configs/experiments/${cfg}.json" \
    --seed 999 \
    --limit 6 || { echo "SMOKE FAILED: ${cfg}" >&2; exit 1; }
done

echo ""
echo "All EU smoke runs finished at $(date)"
echo "Now one ES (Qwen think-mode) smoke run to exercise the thinking path:"
python scripts/run_reasoning_pipeline.py \
  --config configs/experiments/1303_qwen35_9b_marag_e5_rerank5_extractive_mixed_dev.json \
  --seed 999 --limit 6 || { echo "SMOKE FAILED: ES marag" >&2; exit 1; }

echo "Smoke test finished at $(date)"
