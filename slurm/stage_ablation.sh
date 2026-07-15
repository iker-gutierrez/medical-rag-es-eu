#!/bin/bash
#SBATCH --job-name=stage-abl
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/stage_abl_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/stage_abl_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Staged-ablation driver. Runs the correction loop:
#   evaluate -> MeanQ -> rewire mismatched dependents -> re-run them -> repeat,
# then regenerates the result tables and stops.
#
# It requests one GPU because it does the metric evaluation itself (BERTScore on
# GPU); the dependent re-runs it triggers are submitted as SEPARATE Slurm jobs
# (1 GPU/task, %2, GPU-picker, thread-capped) that this driver waits on. So overall
# GPU use stays within the 2-at-a-time convention: this driver's own card plus at
# most... actually the driver is idle-waiting while the rerun array runs, so peak is
# the array's 2. The driver's GPU is only used during its own evaluate step.
#
# Stops BEFORE the reasoning pipelines, which need a manual best-config choice.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "Staged ablation driver started on $(hostname) at $(date)"
python scripts/stage_ablation.py
echo "Staged ablation driver finished at $(date)"
