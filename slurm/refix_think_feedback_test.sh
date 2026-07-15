#!/bin/bash
#SBATCH --job-name=refix-fb-test
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/refix_fb_test_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/refix_fb_test_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "refix-fb-test started on $(hostname) at $(date)"
nvidia-smi || true

python scripts/refix_think_feedback.py --limit-runs 1

echo "refix-fb-test finished at $(date)"
