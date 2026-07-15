#!/bin/bash
#SBATCH --job-name=patch-truncated-updated
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/patch_truncated_updated_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/patch_truncated_updated_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "patch-truncated-updated started on $(hostname) at $(date)"
nvidia-smi || true

echo ""
echo "=== Scanning all _updated runs for per-record truncation flags ==="
python scripts/find_truncated_examples_updated.py

MANIFEST=/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_updated.json

NUM_ENTRIES=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
echo "Found $NUM_ENTRIES run(s) needing a truncation retry."

if [ "$NUM_ENTRIES" -gt 0 ]; then
  echo ""
  echo "=== Running retry generation (max_new_tokens=16000, presence_penalty=0.3) for each truncated run ==="
  # The main runs already generate at max_new_tokens=16000, which covers every
  # genuine answer length (qwen-nothink completed 33/33 with zero truncations).
  # So any record still truncated is a repetition loop, and retrying it without
  # a repetition guard would mostly loop again at 16000 tokens. Apply the
  # proven loop-breaker (presence_penalty=0.3) from the first pass.
  python3 - <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
manifest = json.loads(Path("/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_updated.json").read_text())

for i, entry in enumerate(manifest, start=1):
    config_path = ROOT / entry["rerun_config"]
    config = json.loads(config_path.read_text())
    config["presence_penalty"] = 0.3
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
  echo "=== Patching retry predictions back into original run directories ==="
  python scripts/patch_truncated_predictions.py --manifest "$MANIFEST"

  echo ""
  echo "=== Re-scanning for any records still truncated after the token-limit retry ==="
  echo "(these are likely repetition loops, not genuine length overruns -- retry"
  echo " once more with presence_penalty=0.3, the fix proven for this failure mode"
  echo " earlier this session on sns1064_00200/Latxa)"
  python scripts/find_truncated_examples_updated.py
  REMAINING_AFTER_PASS1=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")

  if [ "$REMAINING_AFTER_PASS1" -gt 0 ]; then
    echo "Found $REMAINING_AFTER_PASS1 run(s) still truncated; retrying with presence_penalty=0.3"
    python3 - <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
manifest = json.loads(Path("/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_updated.json").read_text())

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
    echo "=== Patching presence_penalty retry predictions back in ==="
    python scripts/patch_truncated_predictions.py --manifest "$MANIFEST"
  fi

  echo ""
  echo "=== Re-scanning to verify zero truncations remain ==="
  python scripts/find_truncated_examples_updated.py
  REMAINING=$(python3 -c "import json; print(len(json.load(open('$MANIFEST'))))")
  if [ "$REMAINING" -gt 0 ]; then
    echo "WARNING: $REMAINING run(s) still have truncated records after both retry passes." >&2
    exit 1
  fi
  echo "Confirmed: zero truncated records remain across all _updated runs."
else
  echo "No truncated records found; nothing to patch."
fi

echo ""
echo "patch-truncated-updated finished at $(date)"
