#!/bin/bash
#SBATCH --job-name=reason-eu-latxa1-marag
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_eu_latxa1_marag_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reason_eu_latxa1_marag_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# MA-RAG, Latxa, e5 top-1, 3 seeds. Split from reasoning_eu_latxa_topk1_seeded.sh
# because this one depends on calib_marag_eu_latxa_topk1 (job 9769) having already
# set 1323's conflict_threshold_open via scripts/set_calibrated_thresholds.py --
# submitting it before that would run with threshold=null and crash on float(None).

set -euo pipefail

CONFIG="1323_latxa_llama31_8b_marag_e5_topk1_extractive_mixed_eu_dev"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "reason-eu-latxa1-marag task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "configs/experiments/${CONFIG}.json" \
  --seed "${SEED}"

echo "reason-eu-latxa1-marag task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
