#!/bin/bash
#SBATCH --job-name=medrag-index-eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/index_eu_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/index_eu_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Basque indexing started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

echo "=== SNS1064 EU index ==="
python scripts/build_retrieval_index.py \
  --input data/processed/sns1064_eu/train.jsonl \
  --output-dir models/retrieval/sns1064_eu_train_multilingual_e5_large \
  --backend dense \
  --model intfloat/multilingual-e5-large \
  --batch-size 16 \
  --language eu

echo "=== CasiMedicos EU index ==="
python scripts/build_retrieval_index.py \
  --input data/processed/casimedicos_eu/train.jsonl \
  --output-dir models/retrieval/casimedicos_eu_train_multilingual_e5_large \
  --backend dense \
  --model intfloat/multilingual-e5-large \
  --batch-size 16 \
  --language eu

echo "=== SNS1064+CasiMedicos EU combined index ==="
python scripts/build_retrieval_index.py \
  --input data/processed/sns1064_casimedicos_eu/train.jsonl \
  --output-dir models/retrieval/sns1064_casimedicos_eu_train_multilingual_e5_large \
  --backend dense \
  --model intfloat/multilingual-e5-large \
  --batch-size 16 \
  --language eu

echo "Basque indexing finished at $(date)"
