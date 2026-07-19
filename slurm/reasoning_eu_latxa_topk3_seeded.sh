#!/bin/bash
#SBATCH --job-name=reason-eu-latxa3
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_eu_latxa3_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_eu_latxa3_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Rerun of Latxa's Basque reasoning pipelines on retrieve top-3, replacing the
# topk1 runs of reasoning_eu_latxa_topk1_seeded.sh / reasoning_eu_latxa_marag_seeded.sh:
# once MC-accuracy was correctly included in MeanQ (sec:translation-artefact), the
# single-pass RAG winner for Latxa moved from retrieve top1 to retrieve top3 (same
# as Llama, which already runs its reasoning pipelines on top3 -- configs
# 1310-1313). MA-RAG (1327) is included this time: its conflict_threshold_open is
# now calibrated (calib_marag_eu_latxa_topk3, job 10370 + set_calibrated_thresholds.py),
# unlike the topk1 rerun where it had to be split into its own follow-up array.
# MedCoT-RAG (1343) is NOT part of this array: its causal-aware retrieval is fixed
# by the paper it's faithful to, not by the single-pass ablation winner.
#
# 4 configs (structured_cot, thought_rag, thought_rag_iter, marag) x 3 seeds = 12 runs.
#
# ONE GPU PER TASK, TWO TASKS AT A TIME (%2) -- sec/sample is a measured quantity
# these pipelines report, so latency must not be perturbed by extra concurrency or
# a shared GPU.

set -euo pipefail

CONFIGS=(
  1324_latxa_llama31_8b_structured_cot_e5_topk3_extractive_mixed_eu_dev
  1325_latxa_llama31_8b_thought_rag_e5_topk3_extractive_mixed_eu_dev
  1326_latxa_llama31_8b_thought_rag_iter_e5_topk3_extractive_mixed_eu_dev
  1327_latxa_llama31_8b_marag_e5_topk3_extractive_mixed_eu_dev
)
SEEDS=(42 43 44)

CFG_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
CONFIG="${CONFIGS[$CFG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "reason-eu-latxa3 task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "reason-eu-latxa3 task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
