#!/bin/bash
#SBATCH --job-name=smoke-50pct
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/smoke_50pct_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/smoke_50pct_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# Only the single most expensive sample in the most expensive config: qwen_think's
# casimedicos_576, the longest genuinely-clean (non-looped) record found across
# ALL completed v1 runs of any family (13,788 true observed thinking-tokens,
# verified by reading the full 50,086-char text -- coherent differential-
# diagnosis deliberation, correctly closes </think>, clean final answer).
# This is the only family/config where max_model_len (31744) and max_new_tokens
# (22450) exceed the no_think families' shared 16384/1750, so it's the single
# tightest-margin case worth confirming.
echo ""
echo "=== Running smoke_50pct_qwen_think ==="
python scripts/run_generation_from_config.py --config "configs/experiments/smoke_50pct_qwen_think.json"

echo ""
echo "=== Summary: confirm no truncation under the new 50%-margin budget ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
p = ROOT / "experiments" / "runs" / "smoke_50pct_qwen_think" / "predictions.jsonl"
if not p.exists():
    print("NO OUTPUT FOUND")
else:
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    for r in recs:
        tc = r.get("token_counts", {})
        tflags = r.get("truncation", {})
        text = r.get("prediction_text") or ""
        print(f"id={r['id']}")
        print(f"  finish_reasons(initial/feedback)={tflags.get('initial_finish_reason')}/{tflags.get('feedback_finish_reason')}")
        print(f"  truncated_flags={tflags}")
        print(f"  token_counts={tc}")
        print(f"  final_len_chars={len(text)}")
        print(f"  tail: {text[-150:]!r}")
PYEOF
