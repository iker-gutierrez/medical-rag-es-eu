#!/bin/bash
#SBATCH --job-name=eu-index
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eu_index_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eu_index_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "EU index rebuild started on $(hostname) at $(date)"

# The retranslation (jobs 10193-10195) fixed the translation-truncation
# artefact for real this time: 0 truncated fields across train/dev/test
# (was 109, then 14, now 0 -- see scripts/translate_to_basque.py's history
# for the environment bug, tabular-text bug, and token-budget-gate bug this
# took to actually close). The retrieval indices are built from train, which
# is fully clean.
#
# The gate below is scoped to train+dev only, deliberately excluding test:
# casimedicos_eu/test.jsonl legitimately has 8 more records than
# casimedicos/test.jsonl (their Spanish originals no longer exist anywhere
# in the current corpus, confirmed by searching casimedicos/all.jsonl too --
# not translator hallucination, just orphaned from an earlier ES test-split
# regeneration). Per explicit instruction the fuller 125-record Basque test
# set is kept rather than cut down to match the smaller 117-record Spanish
# one, so the id-set-mismatch check on test is expected to keep "failing"
# and must not block the index rebuild.
#
# Index building embeds the whole corpus and is genuinely GPU-bound, unlike the tiny
# Marian translator. This job requests TWO GPUs (gres=gpu:2) and builds the two
# single-domain indices in parallel, one per card, then the combined index. Latency
# is not being measured here, so parallelism is free.

echo ""
echo "=== integrity gate (train+dev only; test's known, accepted id-count difference is excluded) ==="
python scripts/check_translation_integrity.py --splits train dev

build_index () {
  local gpu="$1" name="$2"
  echo "[GPU ${gpu}] building index for ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/build_retrieval_index.py \
    --input "data/processed/${name}/train.jsonl" \
    --output-dir "models/retrieval/${name}_train_multilingual_e5_large" \
    --model intfloat/multilingual-e5-large \
    --backend dense \
    --language eu
}

echo ""
echo "=== two single-domain indices, one per GPU, in parallel ==="
build_index 0 sns1064_eu &
PID0=$!
build_index 1 casimedicos_eu &
PID1=$!
wait $PID0
wait $PID1

echo ""
echo "=== combined index (GPU 0) ==="
build_index 0 sns1064_casimedicos_eu

echo ""
echo "EU index rebuild finished at $(date)"
