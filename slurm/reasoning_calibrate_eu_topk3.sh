#!/bin/bash
#SBATCH --job-name=calib-eu-topk3
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_eu_topk3_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/calib_eu_topk3_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "EU (e5 top 3) recalibration started on $(hostname) at $(date)"

# The EU retrieval changed from rerank-top-5 to e5-top-3, so the earlier calibration
# is stale: the candidates -- and hence the conflict distribution the open-answer
# threshold is read off -- depend on which evidence the solver was shown.
for cfg in calib_marag_eu_topk3 calib_scot_eu_topk3; do
  echo ""
  echo "######## ${cfg}"
  python scripts/run_reasoning_pipeline.py --config "configs/experiments/${cfg}.json" \
    || { echo "CALIB FAILED: ${cfg}" >&2; exit 1; }
done

echo ""
echo "EU recalibration finished at $(date)"
