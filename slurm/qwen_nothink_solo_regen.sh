#!/bin/bash
#SBATCH --job-name=qwen-nothink-solo-regen
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen_nothink_solo_regen_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen_nothink_solo_regen_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "qwen-nothink-solo-regen started on $(hostname) at $(date)"
nvidia-smi || true

# This regeneration exists to equalize execution conditions across model
# families for the cost/timing metrics: the previous qwen-nothink pass ran
# concurrently with the qwen-think task (shared CPU/memory bandwidth on this
# single-node cluster, measured 1.25-1.75x slowdown on no_think configs),
# while mistral/llama/latxa run solo. Rerunning qwen-nothink solo puts all
# four no_think families under the same conditions.
echo "Removing prior (contended) qwen-nothink _updated runs..."
rm -rf experiments/runs/11[23]*_qwen35_9b_*no_think*_updated_seed*

python scripts/run_casionly_regeneration.py --family qwen-nothink --split mixed

echo "qwen-nothink-solo-regen finished at $(date)"
