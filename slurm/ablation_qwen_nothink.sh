#!/bin/bash
#SBATCH --job-name=abl-qwen_nothink
#SBATCH --array=0-32%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_qwen_nothink_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_qwen_nothink_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Seeded ablation re-run -- qwen_nothink (11 configs x 3 seeds = 33 tasks).
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
  "1128_qwen35_9b_no_rag_no_think_extractive_mixed_dev 42"
  "1128_qwen35_9b_no_rag_no_think_extractive_mixed_dev 43"
  "1128_qwen35_9b_no_rag_no_think_extractive_mixed_dev 44"
  "1129_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev 42"
  "1129_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev 43"
  "1129_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev 44"
  "1130_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev 42"
  "1130_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev 43"
  "1130_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev 44"
  "1131_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev 42"
  "1131_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev 43"
  "1131_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev 44"
  "1132_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev 42"
  "1132_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev 43"
  "1132_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev 44"
  "1133_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev 42"
  "1133_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev 43"
  "1133_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev 44"
  "1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev 42"
  "1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev 43"
  "1134_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev 44"
  "1135_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev 42"
  "1135_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev 43"
  "1135_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev 44"
  "1136_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev 42"
  "1136_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev 43"
  "1136_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev 44"
  "1137_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev 42"
  "1137_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev 43"
  "1137_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev 44"
  "1138_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev 42"
  "1138_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev 43"
  "1138_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev 44"
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

echo "abl-qwen_nothink task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"
nvidia-smi || true

scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "abl-qwen_nothink task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
