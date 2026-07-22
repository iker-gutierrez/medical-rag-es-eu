#!/bin/bash
#SBATCH --job-name=full-corpus-index
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/full_corpus_index_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/full_corpus_index_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Rebuilds every retrieval index (ES + EU, all three corpora) from the FULL
# corpus (train+dev+test), not train-only. Every query now passes
# exclude_id=record["id"] (see run_generation_experiment.py, already updated),
# so a query's own document is always dropped at query time -- the design
# relies on exclude_id, not on the index being restricted to a disjoint split.
#
# Why the change: the previous train-only design under-used the corpus (dev
# and test's own evidence passages were never retrievable, even for OTHER
# queries that could have genuinely benefited from them) and its
# "self-retrieval exclusion" justification didn't hold empirically -- train
# and dev/test instance ids never overlapped, so exclude_id was a no-op on
# the actual ablation-grid queries. All 6 corpora were confirmed to have
# globally unique ids across train/dev/test before this rebuild (no id
# collisions to worry about).
#
# New indices are written under a new _full_ directory name (not overwriting
# the old _train_ ones in place) so that repointing every config's
# retrieval_index field is a single scripted find-replace across the config
# directory, not 1182 individual manual edits, and so a rollback is possible
# by just reverting the config field.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "Full-corpus index rebuild started on $(hostname) at $(date)"

echo ""
echo "=== integrity gate (train+dev only for EU; test's known, accepted casimedicos id-count difference is excluded -- see sec:translation-artefact) ==="
python scripts/check_translation_integrity.py --splits train dev

build_index () {
  local gpu="$1" name="$2"
  echo "[GPU ${gpu}] building full-corpus index for ${name}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/build_retrieval_index.py \
    --input "data/processed/${name}/train.jsonl" \
            "data/processed/${name}/dev.jsonl" \
            "data/processed/${name}/test.jsonl" \
    --output-dir "models/retrieval/${name}_full_multilingual_e5_large" \
    --model intfloat/multilingual-e5-large \
    --backend dense \
    --language "$3"
}

echo ""
echo "=== ES single-domain indices, in parallel ==="
build_index 0 sns1064 es &
PID0=$!
build_index 1 casimedicos es &
PID1=$!
wait $PID0
wait $PID1

echo ""
echo "=== EU single-domain indices, in parallel ==="
build_index 0 sns1064_eu eu &
PID0=$!
build_index 1 casimedicos_eu eu &
PID1=$!
wait $PID0
wait $PID1

echo ""
echo "=== combined indices (sequential, GPU 0) ==="
build_index 0 sns1064_casimedicos es
build_index 0 sns1064_casimedicos_eu eu

echo ""
echo "Full-corpus index rebuild finished at $(date)"
