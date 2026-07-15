#!/bin/bash
#SBATCH --job-name=reeval-report
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reeval_report_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reeval_report_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "reeval-report started on $(hostname) at $(date)"
nvidia-smi || true

echo ""
echo "=== Re-evaluating patched Mistral/Qwen-no_think/Llama/Latxa runs ==="
python scripts/reeval_patched_runs.py

echo ""
echo "=== Regenerating ES ablation report (folds in corrected think-mode + patched metrics) ==="
python scripts/write_mixed_es_seed_summary.py

echo ""
echo "reeval-report finished at $(date)"
