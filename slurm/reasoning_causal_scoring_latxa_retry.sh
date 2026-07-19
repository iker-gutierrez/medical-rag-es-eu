#!/bin/bash
#SBATCH --job-name=reason-causal-latxa
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_causal_latxa_retry_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_causal_latxa_retry_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Retry of config 1343 (Latxa structured_cot, causal-aware retrieval) after
# fixing a real bug: src/medical_rag_thesis/run_logging.py's Tee class (used to
# duplicate stdout/stderr to per-run log files) had no fileno() method, and
# vLLM's engine-core subprocess calls sys.stdout.fileno() during distributed
# group setup (GroupCoordinator.suppress_stdout) -- deterministic
# AttributeError, all 3 seeds, all 3 pick_free_gpu.sh attempts each, in the
# original job (10150, tasks 9-11). 1340 and 1341 (Qwen) happened not to hit
# that code path and completed fine; only Latxa needs rerunning.
#
# ONE GPU PER TASK, TWO TASKS AT A TIME (%2), same convention as every other
# reasoning-pipeline run in this thesis.

set -euo pipefail

CONFIG=1343_latxa_llama31_8b_structured_cot_causal_extractive_mixed_eu_dev
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "reason-causal-latxa-retry task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"
nvidia-smi || true

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "reason-causal-latxa-retry task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
