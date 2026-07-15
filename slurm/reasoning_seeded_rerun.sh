#!/bin/bash
#SBATCH --job-name=reason-rerun
#SBATCH --array=0-23%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_rerun_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_rerun_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Full seeded re-run of the four reasoning pipelines: 8 configs (4 pipelines x 2
# languages) x 3 seeds = 24 runs.
#
# ONE GPU PER TASK, TWO TASKS AT A TIME (%2), exactly like the ablation. This is
# deliberate and must not change: the reasoning pipelines report sec/sample, so
# their latency is a measured quantity. Giving a task a second GPU, or co-scheduling
# more than two, would change the numbers the thesis reports. Evaluation may use two
# GPUs (see eval_ablation_rerun.sh) because it does not measure latency; generation
# may not.
#
# These runs supersede the earlier reasoning runs, which predate three fixes: the
# seed reaching vLLM (so the three seeds are now independent), the query encoder
# pinned to CPU (so the Basque OOM cannot recur), and the corrected Basque
# translation + rebuilt indices.

set -euo pipefail

CONFIGS=(
  1300_qwen35_9b_structured_cot_e5_rerank5_extractive_mixed_dev
  1301_qwen35_9b_thought_rag_e5_rerank5_extractive_mixed_dev
  1302_qwen35_9b_thought_rag_iter_e5_rerank5_extractive_mixed_dev
  1303_qwen35_9b_marag_e5_rerank5_extractive_mixed_dev
  1310_llama31_8b_structured_cot_e5_topk3_extractive_mixed_eu_dev
  1311_llama31_8b_thought_rag_e5_topk3_extractive_mixed_eu_dev
  1312_llama31_8b_thought_rag_iter_e5_topk3_extractive_mixed_eu_dev
  1313_llama31_8b_marag_e5_topk3_extractive_mixed_eu_dev
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
# Cap CPU threads: retrieval (e5) and reranking (cross-encoder) run on CPU with
# torch, which by default spawns one thread per core -- and two tasks run at once
# (%2), on a node shared with other users. Left uncapped this oversubscribes the
# cores and the CPU stages thrash (observed: 27 cores busy, ~zero progress).
# A small cap fixes it and changes no outputs.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "reason-rerun task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"
nvidia-smi || true

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "reason-rerun task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
