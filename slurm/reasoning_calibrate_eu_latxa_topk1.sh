#!/bin/bash
#SBATCH --job-name=calib-eu-latxa-top1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_eu_latxa_top1_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_eu_latxa_top1_%j.err
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

echo "EU Latxa (e5 top 1) MA-RAG conflict-threshold calibration started on $(hostname) at $(date)"

# Latxa's ablation winner is e5 top-1, not top-3 like Llama's -- the earlier EU
# calibration (calib_marag_eu_topk3) is for a different model AND a different
# retrieval depth, so it does not apply here. This calibrates fresh: rounds=1,
# thresholds pinned above 1.0 so nothing settles, every record reports its
# actual round-1 conflict score, read off afterward to set the real threshold.
for cfg in calib_marag_eu_latxa_topk1 calib_scot_eu_latxa_topk1; do
  echo ""
  echo "######## ${cfg}"
  python scripts/run_reasoning_pipeline.py --config "configs/experiments/${cfg}.json" \
    || { echo "CALIB FAILED: ${cfg}" >&2; exit 1; }
done

echo ""
echo "EU Latxa top-1 calibration finished at $(date)"
