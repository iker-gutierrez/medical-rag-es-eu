#!/bin/bash
#SBATCH --job-name=calib-es-nothink
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_es_nothink_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_es_nothink_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "ES Qwen no-think MA-RAG conflict-threshold calibration started on $(hostname) at $(date)"

# The reasoning-pipeline base switched from Qwen3.5-9B think+rerank5 to
# no-think+rerank5 (no-think MeanQ is within noise of think's row-8 winner but at
# ~1/3 the cost). The existing calib_marag_es threshold (0.093) was calibrated on
# think-mode candidates -- no-think produces shorter, more consistent answers, so
# the conflict-score distribution is different and needs its own calibration.
for cfg in calib_marag_es_nothink calib_scot_es_nothink; do
  echo ""
  echo "######## ${cfg}"
  python scripts/run_reasoning_pipeline.py --config "configs/experiments/${cfg}.json" \
    || { echo "CALIB FAILED: ${cfg}" >&2; exit 1; }
done

echo ""
echo "ES no-think calibration finished at $(date)"
