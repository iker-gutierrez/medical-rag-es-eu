#!/bin/bash
#SBATCH --job-name=calib-eu-latxa-topk3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_eu_latxa_topk3_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_eu_latxa_topk3_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "EU Latxa (e5 top 3) recalibration started on $(hostname) at $(date)"

# Latxa's single-pass RAG winner moved from retrieve top1 to retrieve top3 once
# MC-accuracy was correctly included in MeanQ (sec:translation-artefact). MA-RAG's
# open-answer conflict threshold is calibrated off the round-1 candidate
# disagreement, which depends on which evidence the solver was shown -- the old
# topk1 calibration (calib_marag_eu_latxa_topk1) does not transfer to topk3
# retrieval, mirroring the same dependency documented in
# reasoning_calibrate_eu_topk3.sh for Llama's own topk3 recalibration.
for cfg in calib_marag_eu_latxa_topk3 calib_scot_eu_latxa_topk3; do
  echo ""
  echo "######## ${cfg}"
  python scripts/run_reasoning_pipeline.py --config "configs/experiments/${cfg}.json" \
    || { echo "CALIB FAILED: ${cfg}" >&2; exit 1; }
done

echo ""
echo "EU Latxa recalibration finished at $(date)"
