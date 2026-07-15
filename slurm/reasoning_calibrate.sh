#!/bin/bash
#SBATCH --job-name=reason-calib
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reasoning_calib_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reasoning_calib_%j.err
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

echo "Calibration started on $(hostname) at $(date)"

# Two questions, answered on a 24-record stratified sample (12 SNS1064 open-answer
# + 12 CasiMedicos multiple-choice, so BOTH conflict signals get exercised -- the
# dev file is source-ordered, so a plain --limit only ever sees SNS1064):
#
#   1. calib_marag_*  : rounds=1 with the thresholds pinned above 1.0 so nothing
#                       settles -- every record reports its round-1 conflict score.
#                       That distribution is what the open-answer threshold gets
#                       set from, instead of being guessed.
#   2. calib_scot_*   : does the rewritten structured-CoT prompt actually make the
#                       model emit the final answer block? (The first draft did not.)

for cfg in calib_marag_eu calib_scot_eu calib_marag_es calib_scot_es; do
  echo ""
  echo "######## ${cfg}"
  python scripts/run_reasoning_pipeline.py --config "configs/experiments/${cfg}.json" \
    || { echo "CALIB FAILED: ${cfg}" >&2; exit 1; }
done

echo ""
echo "Calibration finished at $(date)"
