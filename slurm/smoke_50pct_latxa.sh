#!/bin/bash
#SBATCH --job-name=smoke-latxa
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/smoke_latxa_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/smoke_latxa_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

START=$(date +%s)
python scripts/run_generation_from_config.py --config "configs/experiments/smoke_50pct_latxa.json"
END=$(date +%s)

echo ""
echo "=== latxa: wall-clock $((END - START))s ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
p = ROOT / "experiments" / "runs" / "smoke_50pct_latxa" / "predictions.jsonl"
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
PYEOF
