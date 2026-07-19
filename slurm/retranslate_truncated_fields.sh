#!/bin/bash
#SBATCH --job-name=retranslate-truncated
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/retranslate_truncated_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/retranslate_truncated_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Targeted fix for sec:translation-artefact: scripts/translate_to_basque.py's
# chunker only split on paragraph breaks, so a long passage with no internal
# break was passed to MarianMT whole and silently truncated at 512 tokens.
# Fixed to pack sentences into token-budgeted groups for any paragraph still
# over the character guard, so no group is both too long AND too short/
# decontextualized for the model. This retranslates ONLY the 109 fields that
# scripts/check_translation_integrity.py actually flags as truncated (ratio
# < 0.30), not the full corpus -- the ~4,370 already-correct fields are
# untouched, since the chunking change only affects behaviour on paragraphs
# that were too long to begin with.
#
# CRITICAL: this environment's default `transformers` (5.10.2, from the
# activated venv) breaks HiTZ/medical_es-eu's generation silently -- every
# translation call, even of a single trivial sentence, produces incoherent
# word-salad output with no error. Confirmed by direct comparison: identical
# input, identical code, only the transformers version and model path
# differed. Must use the vendored 4.26.1 (matching slurm/translate_eu.sh)
# and the pinned local model snapshot, not the bare "HiTZ/medical_es-eu" Hub
# name, which can resolve to a different revision.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/home/igutierrez134/med_rag_thesis/.vendor/transformers426:${PYTHONPATH:-}"

EU_TRANSLATION_MODEL="/home/igutierrez134/.cache/huggingface/models--HiTZ--medical_es-eu/snapshots/38899b3feda911b50b6f7c9a380ba420ff99df65"

echo "Targeted Basque retranslation started on $(hostname) at $(date)"

echo ""
echo "=== translation smoke test (must run BEFORE touching any real data) ==="
python - <<PYEOF
import sys
sys.path.insert(0, "scripts")
from translate_to_basque import load_model, translate_batch

tokenizer, model = load_model("$EU_TRANSLATION_MODEL", "cuda")
texts = ["El paciente presenta dolor abdominal.", "Se recomienda tratamiento con metformina."]
outputs = translate_batch(texts, tokenizer, model, "cuda")
print(outputs)
for src, out in zip(texts, outputs):
    if len(out) > len(src) * 4:
        raise SystemExit(f"Smoke test failed: output ballooned ({len(src)} -> {len(out)} chars)")
bad_fragments = ["Kudeaketa", "askotariko", "zehaztea desager", "Kontsumo abiadura"]
if any(frag in out for out in outputs for frag in bad_fragments):
    raise SystemExit("Smoke test failed: output looks like the known degenerate-generation pattern")
print("Smoke test passed: translation output is coherent.")
PYEOF

# 1. Preserve the truncated data rather than overwrite it: the manuscript
#    documents the artefact, and the old files are the evidence for it.
for d in sns1064_eu casimedicos_eu sns1064_casimedicos_eu; do
  if [[ -d "data/processed/${d}" && ! -d "data/processed/${d}_pre_retranslation_backup" ]]; then
    cp -r "data/processed/${d}" "data/processed/${d}_pre_retranslation_backup"
    echo "backed up data/processed/${d} -> ${d}_pre_retranslation_backup"
  fi
done

# 2. Retranslate exactly the flagged fields in sns1064_eu and casimedicos_eu.
echo ""
echo "=== retranslating flagged fields ==="
python scripts/retranslate_truncated_fields.py --model "$EU_TRANSLATION_MODEL"

# 3. Rebuild the combined Basque dataset from the patched parts.
echo ""
echo "=== rebuilding combined sns1064_casimedicos_eu ==="
for split in train dev test; do
  out="data/processed/sns1064_casimedicos_eu/${split}.jsonl"
  mkdir -p "$(dirname "$out")"
  cat "data/processed/sns1064_eu/${split}.jsonl" \
      "data/processed/casimedicos_eu/${split}.jsonl" > "$out"
  echo "rebuilt ${out} ($(wc -l < "$out") records)"
done

# 4. Verify: no field may still be truncated relative to its Spanish source.
echo ""
echo "=== verification ==="
python scripts/check_translation_integrity.py

# 5. The retrieval index is built from the (previously truncated) train
#    split, so it must be rebuilt or the Basque retriever keeps serving the
#    old, cut-off passages.
echo ""
echo "=== rebuilding Basque retrieval indices ==="
for name in sns1064_eu casimedicos_eu sns1064_casimedicos_eu; do
  python scripts/build_retrieval_index.py \
    --input "data/processed/${name}/train.jsonl" \
    --output-dir "models/retrieval/${name}_train_multilingual_e5_large" \
    --model intfloat/multilingual-e5-large \
    --backend dense \
    --language eu
done

echo ""
echo "Targeted Basque retranslation finished at $(date)"
