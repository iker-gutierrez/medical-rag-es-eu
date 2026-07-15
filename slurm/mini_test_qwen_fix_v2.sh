#!/bin/bash
#SBATCH --job-name=qwen-mini-v2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen_mini_test_v2_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen_mini_test_v2_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "qwen-mini-v2 started on $(hostname) at $(date)"
nvidia-smi || true

# Config is inside the project so parents[2] resolves correctly
python scripts/run_generation_from_config.py \
  --config configs/experiments/mini_test_qwen_fix.json

echo "=== MINI-RUN RESULTS ==="
python3 - << 'PYEOF'
import json

try:
    lines = open('/tmp/qwen_mini_predictions.jsonl').readlines()
except FileNotFoundError:
    print("ERROR: predictions file not found")
    exit(1)

print(f"Total predictions: {len(lines)}")
for i, line in enumerate(lines):
    d = json.loads(line)
    init = d.get('initial_prediction_text', '') or d.get('parsed_initial_prediction', '')
    pred = d.get('prediction_text', '') or d.get('parsed_prediction', '')
    init_words = init.split()
    pred_words = pred.split()
    init_max = max((len(w) for w in init_words), default=0)
    pred_max = max((len(w) for w in pred_words), default=0)
    init_ok = "OK" if init_max <= 30 else "BAD"
    pred_ok = "OK" if pred_max <= 30 else "BAD"
    print(f"[{i}] INIT [{init_ok}, max_word={init_max}]: {repr(init[:120])}")
    print(f"     SF   [{pred_ok}, max_word={pred_max}]: {repr(pred[:120])}")
    print()
PYEOF

echo "qwen-mini-v2 finished at $(date)"
