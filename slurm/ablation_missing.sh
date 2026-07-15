#!/bin/bash
#SBATCH --job-name=abl-missing
#SBATCH --array=0-6%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_missing_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/abl_missing_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Re-run of the ablation runs that FAILED to vLLM engine-init when the shared GPUs
# were full (ConstrainDevices=no, so Slurm's GPU assignment is not enforced). Uses
# pick_free_gpu.sh to land only on a genuinely-idle card, keeping gpu_memory
# utilization at 0.90 (so the KV cache and measured latency are unchanged).

set -euo pipefail

mapfile -t TASKS < experiments/missing_runs.txt
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

echo "abl-missing task ${SLURM_ARRAY_TASK_ID}: CONFIG=${CONFIG} SEED=${SEED} on $(hostname) at $(date)"

scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "abl-missing task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
