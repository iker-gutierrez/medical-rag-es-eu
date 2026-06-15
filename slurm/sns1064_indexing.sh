#!/bin/bash
#SBATCH --job-name=medrag-sns-index
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/sns1064_indexing_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/sns1064_indexing_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p experiments/slurm_logs

echo "SNS1064 indexing started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nvidia-smi || true

python scripts/build_retrieval_index.py \
  --input data/processed/sns1064/train.jsonl \
  --output-dir models/retrieval/sns1064_train_multilingual_minilm \
  --backend dense \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

python scripts/build_retrieval_index.py \
  --input data/processed/sns1064/train.jsonl \
  --output-dir models/retrieval/sns1064_train_tfidf \
  --backend tfidf

echo "SNS1064 indexing finished at $(date)"
