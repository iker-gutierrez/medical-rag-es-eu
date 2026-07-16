#!/bin/bash
#SBATCH --job-name=reason-es-nothink
#SBATCH --array=0-8%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_es_nothink_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_es_nothink_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Seeded reasoning-pipeline runs for Spanish with Qwen3.5-9B no-think + rerank5
# (the chosen base: MeanQ within noise of think mode's row-8 winner, ~1/3 the cost).
# 3 configs (structured_cot, thought_rag, thought_rag_iter) x 3 seeds = 9 runs.
# MA-RAG (1333) is NOT in this array: its conflict_threshold_open is still null
# pending calibration (calib_marag_es_nothink) -- see reasoning_es_nothink_marag_seeded.sh.
#
# ONE GPU PER TASK, TWO TASKS AT A TIME (%2) -- sec/sample is a measured quantity
# these pipelines report, so latency must not be perturbed by extra concurrency.

set -euo pipefail

CONFIGS=(
  1330_qwen35_9b_structured_cot_e5_rerank5_no_think_extractive_mixed_dev
  1331_qwen35_9b_thought_rag_e5_rerank5_no_think_extractive_mixed_dev
  1332_qwen35_9b_thought_rag_iter_e5_rerank5_no_think_extractive_mixed_dev
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

echo "reason-es-nothink task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "reason-es-nothink task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
