#!/bin/bash
#SBATCH --job-name=trunc-retry-regen
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/truncation_retry_regen_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/truncation_retry_regen_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# Regenerates ONLY the 50 records still genuinely broken after the earlier
# offline loop-fix (across Llama/Latxa/Qwen-nothink), using
# --max-truncation-retries=3 (real-time, per-record retry inside
# run_generation_experiment.py -- see generate_batch_with_retry) so each
# gets up to 3 fresh extra attempts if it hits finish_reason
# length/repetition again. Each config's own real retrieval/reranking
# settings are used, just with a tiny input file (only the broken record
# ids for that config+seed) instead of the full 126-record set, so no
# unaffected record is ever touched.

SCRATCH="/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_retry_regen"

python3 - <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/igutierrez134/med_rag_thesis/scripts")
from run_generation_from_config import build_command

SCRATCH = Path("/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_retry_regen")
manifest = json.loads((SCRATCH / "manifest.json").read_text())

for i, entry in enumerate(manifest, start=1):
    print(f"\n=== [{i}/{len(manifest)}] {entry['config_base']} seed={entry['seed']} ({len(entry['record_ids'])} records) ===")
    cmd = build_command(
        Path(entry["retry_config_path"]),
        dry_run=False,
        save_prompts=False,
        seed_override=entry["seed_int"],
    )
    result = subprocess.run(cmd, cwd="/home/igutierrez134/med_rag_thesis")
    if result.returncode != 0:
        print(f"FAILED: {entry['config_base']} seed={entry['seed']}", file=sys.stderr)
PYEOF

echo ""
echo "Truncation-retry regeneration complete."
