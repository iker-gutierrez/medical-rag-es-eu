#!/bin/bash
#SBATCH --job-name=reason-causal
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_causal_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_causal_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# MedCoT-RAG causal-aware retrieval scoring (sec:reasoning-pipelines,
# src/medical_rag_thesis/causal_scoring.py): 4 structured_cot configs (one per
# model/language pairing already used for the non-causal structured_cot runs)
# x 3 seeds = 12 runs. These are NEW experiment IDs (1340-1343) with their own
# output paths -- they do not touch or overwrite the existing 1300/1310/1320/1330
# structured_cot predictions, which stay as the "our retrieval + their CoT shape"
# variant reported alongside these.
#
# ONE GPU PER TASK, TWO TASKS AT A TIME (%2), matching every other reasoning-
# pipeline and ablation run in this thesis (see reasoning_seeded_rerun.sh):
# these runs report sec/sample, so latency is a measured quantity and must not
# be confounded by GPU sharing.

set -euo pipefail

CONFIGS=(
  1340_qwen35_9b_structured_cot_causal_extractive_mixed_dev
  1341_qwen35_9b_structured_cot_causal_no_think_extractive_mixed_dev
  1342_llama31_8b_structured_cot_causal_extractive_mixed_eu_dev
  1343_latxa_llama31_8b_structured_cot_causal_extractive_mixed_eu_dev
)
SEEDS=(42 43 44)

# task -> (config, seed). config-major so a partial array still spans seeds evenly.
CFG_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
CONFIG="${CONFIGS[$CFG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
# Cap CPU threads: retrieval (e5) and causal re-scoring run on CPU with torch,
# which by default spawns one thread per core -- and two tasks run at once
# (%2), on a node shared with other users. See reasoning_seeded_rerun.sh.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "reason-causal task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"
nvidia-smi || true

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "reason-causal task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
