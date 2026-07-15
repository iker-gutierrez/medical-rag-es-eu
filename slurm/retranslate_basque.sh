#!/bin/bash
#SBATCH --job-name=retranslate-eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/retranslate_eu_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/retranslate_eu_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Basque retranslation started on $(hostname) at $(date)"

# Why this rerun exists: the previous Basque data was produced by a translator that
# split only on paragraph breaks and used a CHARACTER threshold as a proxy for
# MarianMT's 512-TOKEN input limit. A long passage with no blank line was passed to
# the model whole and silently truncated. 20.6% of dev and 15.4% of train evidence
# fields lost more than half their content; the worst went from 10,604 characters to
# 1,143. Those truncated fields were the GOLD references the Basque evidence metrics
# were scored against, and (since the index is built from train) also the passages
# the Basque retriever returned.
#
# translate_to_basque.py now packs SENTENCES up to a real token budget, measured
# with the model's own tokeniser. Verified on the full corpus: 2,112 chunks, none
# over 512 tokens, and reassembly is lossless.

# 1. Preserve the truncated data rather than overwrite it: the manuscript documents
#    the artefact, and the old files are the evidence for it.
for d in sns1064_eu casimedicos_eu sns1064_casimedicos_eu; do
  if [[ -d "data/processed/${d}" && ! -d "data/processed/${d}_truncated" ]]; then
    cp -r "data/processed/${d}" "data/processed/${d}_truncated"
    echo "archived data/processed/${d} -> ${d}_truncated"
  fi
done

# 2. Retranslate every split of both source datasets.
for ds in sns1064 casimedicos; do
  for split in train dev test; do
    src="data/processed/${ds}/${split}.jsonl"
    dst="data/processed/${ds}_eu/${split}.jsonl"
    [[ -f "$src" ]] || { echo "SKIP (no source): $src"; continue; }
    echo ""
    echo "### translating ${src} -> ${dst}"
    python scripts/translate_to_basque.py --input "$src" --output "$dst"
  done
done

# 3. Rebuild the combined Basque dataset from the freshly translated parts.
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

# 5. The retrieval index is built from the (previously truncated) train split, so it
#    must be rebuilt or the Basque retriever keeps serving the old, cut-off passages.
echo ""
echo "=== rebuilding Basque retrieval indices ==="
for name in sns1064_eu casimedicos_eu sns1064_casimedicos_eu; do
  python scripts/build_retrieval_index.py \
    --input "data/processed/${name}/train.jsonl" \
    --output "models/retrieval/${name}_train_multilingual_e5_large" \
    --model intfloat/multilingual-e5-large \
    --backend dense \
    --language eu
done

echo ""
echo "Basque retranslation finished at $(date)"
