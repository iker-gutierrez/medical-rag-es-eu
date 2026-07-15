#!/bin/bash
#SBATCH --job-name=patch-truncated-v2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/patch_truncated_v2_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/patch_truncated_v2_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "patch-truncated-v2 started on $(hostname) at $(date)"
nvidia-smi || true

MANIFEST=/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_v2.json

echo ""
echo "=== Scanning all _v2 runs (all 5 families) for per-record truncation flags ==="
python scripts/find_truncated_examples_v2.py

NUM_ENTRIES=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
echo "Found $NUM_ENTRIES run(s) needing a truncation retry."

if [ "$NUM_ENTRIES" -gt 0 ]; then
  # Unlike the v1 gate, the v2 main pass already generates at the right-sized
  # budget (8000 no_think / 20000 think) from the start, so any record still
  # truncated here is a repetition loop, not a genuine length overrun.
  # Go straight to the proven loop-breaker (presence_penalty=0.3).
  echo ""
  echo "=== Running retry generation (presence_penalty=0.3) for each truncated run ==="
  python3 - <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
manifest = json.loads(Path("/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_v2.json").read_text())

for i, entry in enumerate(manifest, start=1):
    config_path = ROOT / entry["rerun_config"]
    config = json.loads(config_path.read_text())
    config["presence_penalty"] = 0.3
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"[{i}/{len(manifest)}] Retry-generating {entry['run_dir_name']} with presence_penalty=0.3 ({entry['num_truncated_ids']} id(s))")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_generation_from_config.py"),
         "--config", str(config_path)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"FAILED: {entry['run_dir_name']} (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)
PYEOF

  echo ""
  echo "=== Patching retry predictions back into original run directories ==="
  python scripts/patch_truncated_predictions.py --manifest "$MANIFEST"

  echo ""
  echo "=== Re-scanning for stragglers; retry once more with a larger bump if any remain ==="
  python scripts/find_truncated_examples_v2.py
  REMAINING_AFTER_PASS1=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")

  if [ "$REMAINING_AFTER_PASS1" -gt 0 ]; then
    echo "Found $REMAINING_AFTER_PASS1 run(s) still truncated; retrying with presence_penalty=0.3 and a doubled token budget"
    python3 - <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
manifest = json.loads(Path("/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_v2.json").read_text())

for i, entry in enumerate(manifest, start=1):
    config_path = ROOT / entry["rerun_config"]
    config = json.loads(config_path.read_text())
    config["presence_penalty"] = 0.3
    config["max_new_tokens"] = config.get("max_new_tokens", 8000) * 2
    if "feedback_max_new_tokens" in config:
        config["feedback_max_new_tokens"] = config["max_new_tokens"]
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"[{i}/{len(manifest)}] Retry-generating {entry['run_dir_name']} ({entry['num_truncated_ids']} id(s))")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_generation_from_config.py"),
         "--config", str(config_path)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"FAILED: {entry['run_dir_name']} (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)
PYEOF

    echo ""
    echo "=== Patching second retry predictions back in ==="
    python scripts/patch_truncated_predictions.py --manifest "$MANIFEST"
  fi

  echo ""
  echo "=== Re-scanning to verify zero truncations remain ==="
  python scripts/find_truncated_examples_v2.py
  REMAINING=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
  if [ "$REMAINING" -gt 0 ]; then
    echo "WARNING: $REMAINING run(s) still have truncated records after both retry passes." >&2
    exit 1
  fi
  echo "Confirmed: zero truncated records remain across all _v2 runs."
else
  echo "No truncated records found; nothing to patch."
fi

echo ""
echo "patch-truncated-v2 finished at $(date)"
