#!/bin/bash
#SBATCH --job-name=reindex-es
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reindex_spanish_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/reindex_spanish_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Spanish reindexing started on $(hostname) at $(date)"
nvidia-smi || true

echo "=== SNS1064 ==="
python scripts/build_retrieval_index.py \
  --input data/processed/sns1064/train.jsonl \
         data/processed/sns1064/dev.jsonl \
         data/processed/sns1064/test.jsonl \
  --output-dir models/retrieval/sns1064_train_multilingual_e5_large \
  --backend dense \
  --model intfloat/multilingual-e5-large \
  --batch-size 16

echo "=== CasiMedicos ==="
python scripts/build_retrieval_index.py \
  --input data/processed/casimedicos/train.jsonl \
         data/processed/casimedicos/dev.jsonl \
         data/processed/casimedicos/test.jsonl \
  --output-dir models/retrieval/casimedicos_train_multilingual_e5_large \
  --backend dense \
  --model intfloat/multilingual-e5-large \
  --batch-size 16

echo "=== SNS1064+CasiMedicos mixed ==="
python scripts/build_retrieval_index.py \
  --input data/processed/sns1064_casimedicos/train.jsonl \
         data/processed/sns1064_casimedicos/dev.jsonl \
         data/processed/sns1064_casimedicos/test.jsonl \
  --output-dir models/retrieval/sns1064_casimedicos_train_multilingual_e5_large \
  --backend dense \
  --model intfloat/multilingual-e5-large \
  --batch-size 16

echo "Spanish reindexing finished at $(date)"
