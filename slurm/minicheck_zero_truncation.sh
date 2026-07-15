#!/bin/bash
#SBATCH --job-name=minicheck-zero-trunc
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/minicheck_zero_trunc_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/minicheck_zero_trunc_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "minicheck-zero-trunc started on $(hostname) at $(date)"

CONFIGS=(
  minicheck_1138_qwen_nothink
  minicheck_1270_mistral
  minicheck_1048_llama
  minicheck_1059_latxa
)

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "=== Running $cfg (worst-case prompt for its family) ==="
  python scripts/run_generation_from_config.py --config "configs/experiments/${cfg}.json"
done

echo ""
echo "=== Checking truncation flags across all minicheck runs ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

RUNS = [
    "minicheck_1138_qwen_nothink",
    "minicheck_1270_mistral",
    "minicheck_1048_llama",
    "minicheck_1059_latxa",
]
ROOT = Path("/home/igutierrez134/med_rag_thesis")
any_truncated = False
for run in RUNS:
    p = ROOT / "experiments" / "runs" / run / "predictions.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in recs:
        tc = r.get("token_counts", {})
        tflags = r.get("truncation", {})
        truncated = any(v is True for v in tflags.values())
        status = "TRUNCATED" if truncated else "clean"
        print(f"{run}: {r['id']} -> {status} | initial_output_tokens={tc.get('initial_output_tokens')} output_tokens={tc.get('output_tokens')} input_tokens={tc.get('input_tokens')} feedback_input_tokens={tc.get('feedback_input_tokens')}")
        any_truncated = any_truncated or truncated

print()
if any_truncated:
    print("RESULT: at least one minicheck record truncated -- hyperparameters NOT sufficient.")
else:
    print("RESULT: zero truncations across all 4 no_think families' worst-case prompts.")
PYEOF

echo ""
echo "minicheck-zero-trunc finished at $(date)"
