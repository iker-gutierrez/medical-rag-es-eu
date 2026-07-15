#!/bin/bash
#SBATCH --job-name=abl-latxa
#SBATCH --array=0-32%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_latxa_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_latxa_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Seeded ablation re-run -- latxa (11 configs x 3 seeds = 33 tasks).
#
# Supersedes the runs now archived in experiments/runs_v2. Those were generated
# before generation.py passed --seed to vLLM's SamplingParams, so their three
# "seeds" were not independent samples (several are byte-identical) and their
# reported +/-std across seeds is meaningless. Point estimates were fine; the
# error bars were not.
#
# %2 throttles the array to two concurrent GPU jobs, matching the previous
# ablation's resource usage.

set -euo pipefail

TASKS=(
  "1051_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev 42"
  "1051_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev 43"
  "1051_latxa_llama31_8b_no_rag_extractive_mixed_eu_dev 44"
  "1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev 42"
  "1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev 43"
  "1052_latxa_llama31_8b_rag_e5_topk1_extractive_mixed_eu_dev 44"
  "1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev 42"
  "1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev 43"
  "1053_latxa_llama31_8b_rag_e5_topk3_extractive_mixed_eu_dev 44"
  "1054_latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev 42"
  "1054_latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev 43"
  "1054_latxa_llama31_8b_rag_e5_topk5_extractive_mixed_eu_dev 44"
  "1055_latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev 42"
  "1055_latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev 43"
  "1055_latxa_llama31_8b_rag_e5_rerank1_extractive_mixed_eu_dev 44"
  "1056_latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev 42"
  "1056_latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev 43"
  "1056_latxa_llama31_8b_rag_e5_rerank3_extractive_mixed_eu_dev 44"
  "1057_latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev 42"
  "1057_latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev 43"
  "1057_latxa_llama31_8b_rag_e5_rerank5_extractive_mixed_eu_dev 44"
  "1058_latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev 42"
  "1058_latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev 43"
  "1058_latxa_llama31_8b_3shot_no_rag_extractive_mixed_eu_dev 44"
  "1059_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev 42"
  "1059_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev 43"
  "1059_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_mixed_eu_dev 44"
  "1060_latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev 42"
  "1060_latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev 43"
  "1060_latxa_llama31_8b_rag_sns1064_e5_rerank5_extractive_mixed_eu_dev 44"
  "1061_latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev 42"
  "1061_latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev 43"
  "1061_latxa_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev 44"
)

ENTRY="${TASKS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="${ENTRY%% *}"
SEED="${ENTRY##* }"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
# Cap CPU threads: retrieval (e5) and reranking (cross-encoder) run on CPU with
# torch, which by default spawns one thread per core -- and two tasks run at once
# (%2), on a node shared with other users. Left uncapped this oversubscribes the
# cores and the CPU stages thrash (observed: 27 cores busy, ~zero progress).
# A small cap fixes it and changes no outputs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "abl-latxa task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"
nvidia-smi || true

scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "abl-latxa task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
