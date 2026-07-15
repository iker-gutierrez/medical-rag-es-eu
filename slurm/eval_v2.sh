#!/bin/bash
#SBATCH --job-name=eval-v2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_v2_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_v2_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "eval-v2 started on $(hostname) at $(date)"
nvidia-smi || true

echo ""
echo "=== Evaluating v2 mixed-dev runs (auto-splits by source: overall/sns1064/casimedicos) ==="
python - <<'PYEOF'
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
RUNS_DIR = ROOT / "experiments" / "runs"
METRICS_DIR = ROOT / "reports" / "metrics"

def determine_lang(run_dir_name: str) -> str:
    return "eu" if "_eu_dev" in run_dir_name else "es"

run_dirs = sorted(RUNS_DIR.glob("*_v2_seed*"))
print(f"Found {len(run_dirs)} v2 mixed-dev run directories")

for i, run_dir in enumerate(run_dirs, start=1):
    predictions = run_dir / "predictions.jsonl"
    if not predictions.exists():
        print(f"[{i}/{len(run_dirs)}] SKIP (no predictions): {run_dir.name}")
        continue

    out = METRICS_DIR / f"{run_dir.name}.json"
    if out.exists():
        print(f"[{i}/{len(run_dirs)}] SKIP (already evaluated): {run_dir.name}")
        continue

    lang = determine_lang(run_dir.name)
    print(f"[{i}/{len(run_dirs)}] Evaluating {run_dir.name} (lang={lang})")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_predictions.py"),
         "--predictions", str(predictions.relative_to(ROOT)),
         "--output", str(out.relative_to(ROOT)),
         "--semantic-model", "intfloat/multilingual-e5-large",
         "--bertscore-model", "bert-base-multilingual-cased",
         "--bertscore-lang", lang],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"FAILED: {run_dir.name} (exit {result.returncode})", file=sys.stderr)

print("\nAll v2 mixed-dev runs evaluated.")
PYEOF

echo ""
echo "eval-v2 finished at $(date). Metrics are in reports/metrics/*_v2_seed*.json"
echo "(report table generation -- e.g. a v1-vs-v2 comparison table -- is a separate step, not run here)"
