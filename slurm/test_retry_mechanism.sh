#!/bin/bash
#SBATCH --job-name=test-retry
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/test_retry_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/test_retry_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# sns1064_00839 under config 1045 is a known repetition-cut case
# (feedback_finish_reason="repetition", final answer only 528 chars).
# Testing with --max-truncation-retries=2 to confirm the retry mechanism
# actually fires and check whether a fresh sample recovers a complete answer.
python scripts/run_generation_experiment.py \
  --input /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/retry_test_record.jsonl \
  --output experiments/runs/test_retry_mechanism/predictions.jsonl \
  --experiment-name test_retry_mechanism \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-style extractive --language eu \
  --max-new-tokens 1750 --temperature 0.6 --top-p 0.9 --min-p 0.05 --presence-penalty 0.1 \
  --retrieval-index models/retrieval/sns1064_casimedicos_eu_train_multilingual_e5_large \
  --retrieval-top-k 15 \
  --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --reranker-top-k 3 --reranker-device cpu \
  --feedback-max-new-tokens 1750 --max-model-len 13312 \
  --repetition-detection-max-pattern 20 --repetition-detection-min-pattern 1 --repetition-detection-min-count 8 \
  --max-truncation-retries 2 \
  --self-feedback

echo ""
echo "=== Result ==="
python3 -c "
import json
recs = [json.loads(l) for l in open('experiments/runs/test_retry_mechanism/predictions.jsonl')]
r = recs[0]
print('truncation:', r.get('truncation'))
print('final len:', len(r.get('prediction_text') or ''))
print(r.get('prediction_text'))
"
