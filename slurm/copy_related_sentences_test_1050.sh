#!/bin/bash
#SBATCH --job-name=copy-rel-1050
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/copy_related_sentences_test_1050_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/copy_related_sentences_test_1050_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

SCRATCH="/tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/no_copy_prompt_variant"

# Full config 1050, single seed (config's own default 42), but the EU
# extractive prompt's copy-exact-phrases sentence is rephrased to condition
# on SENTENCE-level relevance ("Galderarekin zuzenki erlazionatuta dauden
# esaldiak literalki kopiatu ditzakezu...") rather than DOCUMENT-level
# relevance ("Kopiatu esaldi osoak ... testuingurua galderarekin zuzenean
# lotuta dagoenean") -- via a standalone copy of run_generation_experiment.py
# whose sys.path puts the scratch src/ (edited prompts.py) first, so the
# live production prompts.py is never touched.
python "$SCRATCH/run_generation_experiment_no_copy.py" \
  --input data/processed/sns1064_casimedicos_eu/dev.jsonl \
  --output experiments/runs/copy_related_sentences_test_1050/predictions.jsonl \
  --experiment-name llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-style extractive --language eu \
  --max-new-tokens 1750 --temperature 0.6 --top-p 0.9 --min-p 0.05 --presence-penalty 0.1 \
  --retrieval-index models/retrieval/casimedicos_eu_train_multilingual_e5_large \
  --retrieval-top-k 15 \
  --reranker-model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --reranker-top-k 5 --reranker-device cpu \
  --feedback-max-new-tokens 1750 --max-model-len 13312 \
  --repetition-detection-max-pattern 20 --repetition-detection-min-pattern 1 --repetition-detection-min-count 8 \
  --self-feedback

echo ""
echo "=== Truncation/loop comparison: NEW (sentence-level copy license) vs ORIGINAL (document-level) ==="
python3 - <<'PYEOF'
import json
import sys
sys.path.insert(0, "/home/igutierrez134/med_rag_thesis/scripts")
from fix_truncated_loops_in_place import is_truncated_record, find_loop_cutpoint
from pathlib import Path

ROOT = Path("/home/igutierrez134/med_rag_thesis")
new_path = ROOT / "experiments/runs/copy_related_sentences_test_1050/predictions.jsonl"
orig_path = ROOT / "experiments/runs/1050_llama31_8b_rag_casimedicos_e5_rerank5_extractive_mixed_eu_dev_v2_seed42/predictions.jsonl"

new_recs = [json.loads(l) for l in new_path.read_text().splitlines() if l.strip()]
orig_recs = {r["id"]: r for r in (json.loads(l) for l in orig_path.read_text().splitlines() if l.strip())} if orig_path.exists() else {}

new_truncated = [r for r in new_recs if is_truncated_record(r)]
print(f"NEW (sentence-level copy license): {len(new_truncated)}/{len(new_recs)} truncated")
for r in new_truncated:
    print(f"  {r['id']}: {r.get('truncation')}")

print()
problem_ids = ["sns1064_00089", "sns1064_00369", "sns1064_00213", "sns1064_00290", "sns1064_00200"]
for rid in problem_ids:
    n = next((r for r in new_recs if r["id"] == rid), None)
    o = orig_recs.get(rid)
    print(f"\n=== {rid} ===")
    if n:
        print(f"  NEW: finish={n.get('truncation', {}).get('feedback_finish_reason')} len={len(n.get('prediction_text') or '')}")
        print(f"    tail: {(n.get('prediction_text') or '')[-200:]!r}")
    if o:
        print(f"  ORIGINAL: finish={o.get('truncation', {}).get('feedback_finish_reason')} len={len(o.get('prediction_text') or '')}")
        print(f"    tail: {(o.get('prediction_text') or '')[-200:]!r}")
PYEOF
