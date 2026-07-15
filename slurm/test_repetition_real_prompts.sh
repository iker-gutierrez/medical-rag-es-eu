#!/bin/bash
#SBATCH --job-name=test-repdet-real
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:45:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/test_repdet_real_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/test_repdet_real_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

CONFIGS=(
  loop_test_1268_baseline
  loop_test_1268
  loop_test_1261_baseline
  loop_test_1261
  loop_test_1044_baseline
  loop_test_1044
)

for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "=== Running $cfg ==="
  python scripts/run_generation_from_config.py --config "configs/experiments/${cfg}.json"
done

echo ""
echo "=== Summary: output lengths and truncation flags per case ==="
python3 - <<'PYEOF'
import json
from pathlib import Path

RUNS = [
    ("1268 (Mistral, casimedicos_65)", "loop_test_1268_baseline", "loop_test_1268"),
    ("1261 (Mistral, sns1064_00603)", "loop_test_1261_baseline", "loop_test_1261"),
    ("1044 (Llama, casimedicos_531)", "loop_test_1044_baseline", "loop_test_1044"),
]
ROOT = Path("/home/igutierrez134/med_rag_thesis")

for label, baseline_dir, repdet_dir in RUNS:
    print(f"\n--- {label} ---")
    for tag, run_dir in [("baseline (no repetition_detection)", baseline_dir), ("with repetition_detection", repdet_dir)]:
        p = ROOT / "experiments" / "runs" / run_dir / "predictions.jsonl"
        if not p.exists():
            print(f"  {tag}: NO OUTPUT FOUND")
            continue
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for r in recs:
            text = r.get("initial_prediction_text") or r.get("prediction_text") or ""
            tc = r.get("token_counts", {})
            tflags = r.get("truncation", {})
            print(f"  {tag}: id={r['id']} finish_reasons(initial/feedback)={tflags.get('initial_finish_reason')}/{tflags.get('feedback_finish_reason')} "
                  f"len_chars={len(text)} initial_output_tokens={tc.get('initial_output_tokens')} output_tokens={tc.get('output_tokens')}")
            print(f"    tail: {text[-150:]!r}")
PYEOF
