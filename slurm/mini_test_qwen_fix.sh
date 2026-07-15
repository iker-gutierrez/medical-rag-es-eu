#!/bin/bash
#SBATCH --job-name=qwen-mini-test
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen_mini_test_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen_mini_test_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "qwen-mini-test started on $(hostname) at $(date)"
nvidia-smi || true

python scripts/run_generation_from_config.py --config /tmp/qwen_mini_test.json

echo "=== MINI-RUN RESULTS ==="
python3 - << 'PYEOF'
import json

lines = open('/tmp/qwen_mini_predictions.jsonl').readlines()
print(f"Total predictions: {len(lines)}")
for i, line in enumerate(lines):
    d = json.loads(line)
    ref = d.get('reference_short_answer', '')
    init = d.get('initial_prediction_text', '')[:120]
    pred = d.get('prediction_text', '')[:120]
    init_words = init.split()
    pred_words = pred.split()
    init_max = max((len(w) for w in init_words), default=0)
    pred_max = max((len(w) for w in pred_words), default=0)
    init_ok = "OK" if init_max <= 30 else "BAD"
    pred_ok = "OK" if pred_max <= 30 else "BAD"
    print(f"[{i}] REF: {repr(ref[:50])}")
    print(f"     INIT [{init_ok}]: {repr(init[:100])}")
    print(f"     SF   [{pred_ok}]: {repr(pred[:100])}")
    print()
PYEOF

echo "qwen-mini-test finished at $(date)"
