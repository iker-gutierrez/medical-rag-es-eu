#!/bin/bash
#SBATCH --job-name=reason-eu-latxa1
#SBATCH --array=0-8%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_eu_latxa1_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_eu_latxa1_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Seeded reasoning-pipeline runs for Basque with Latxa on its own ablation winner,
# e5 top-1 -- not the Llama/top-3 configs reasoning_seeded_rerun.sh already covers.
# 3 configs (structured_cot, thought_rag, thought_rag_iter) x 3 seeds = 9 runs.
# MA-RAG (1323) is deliberately NOT in this array: its conflict_threshold_open is
# still null pending calibration (calib_marag_eu_latxa_topk1, job 9769) -- add it
# to a follow-up array once scripts/analyze_conflict_calibration.py reports the
# eu_latxa_topk1 distribution and the threshold is filled in.
#
# ONE GPU PER TASK, TWO TASKS AT A TIME (%2) -- same reason as reasoning_seeded_rerun.sh:
# sec/sample is a measured quantity these pipelines report, so latency must not be
# perturbed by extra concurrency or a shared GPU.

set -euo pipefail

CONFIGS=(
  1320_latxa_llama31_8b_structured_cot_e5_topk1_extractive_mixed_eu_dev
  1321_latxa_llama31_8b_thought_rag_e5_topk1_extractive_mixed_eu_dev
  1322_latxa_llama31_8b_thought_rag_iter_e5_topk1_extractive_mixed_eu_dev
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

echo "reason-eu-latxa1 task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "reason-eu-latxa1 task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
