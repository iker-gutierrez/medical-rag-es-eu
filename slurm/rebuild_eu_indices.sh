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

# The retranslation (job 9379) already wrote all six Basque splits and the three
# combined sets to disk correctly; it only failed at the integrity gate, which now
# passes (the one genuine under-translation is flagged for manual translation, not a
# systematic bug). So translation does NOT need re-running -- only the retrieval
# indices, which are built from the (now corrected) train split.
#
# Index building embeds the whole corpus and is genuinely GPU-bound, unlike the tiny
# Marian translator. This job requests TWO GPUs (gres=gpu:2) and builds the two
# single-domain indices in parallel, one per card, then the combined index. Latency
# is not being measured here, so parallelism is free.

echo ""
echo "=== integrity gate (must pass before rebuilding indices) ==="
python scripts/check_translation_integrity.py

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
