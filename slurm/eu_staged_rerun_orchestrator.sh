#!/bin/bash
#SBATCH --job-name=eu-staged-rerun
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=8GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eu_staged_rerun_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eu_staged_rerun_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Orchestrator only -- no GPU work happens directly in this job. It calls
# scripts/staged_ablation_runner.py, which itself submits and waits on the
# real 1-GPU-per-task, %2-throttled inference arrays (see submit_infer_array
# in that script). This job just needs to stay alive and poll while those
# child jobs run, hence the long time limit and minimal resource request.
#
# Runs the full staged EU ablation (Llama + Latxa, all 11 rows each, MeanQ-
# wired dependencies at inference time) against the corrected corpus and
# rebuilt retrieval indices from the translation-truncation fix
# (sec:translation-artefact). Chained via --dependency=afterok on the index
# rebuild job, so this never starts against stale indices.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "EU staged rerun orchestrator started on $(hostname) at $(date)"

python scripts/staged_ablation_runner.py --models llama31_8b latxa

echo ""
echo "=== regenerating the raw EU summary report (decision tables only; NOT the manuscript tables) ==="
python scripts/write_mixed_eu_seed_summary.py

# STOPS HERE, deliberately. write_result_tables.py's FORCED_REFERENCES and
# write_reasoning_latex_table.py's EU_BASELINE_DESC are hardcoded per-model
# pins (sec:reasoning-pipelines) chosen from the PREVIOUS ablation's MeanQ
# winners. This rerun, on the corrected corpus, may pick a different winner
# (the pre-fix dry run already showed "retrieve top 3" beating the currently
# pinned "retrieve top 1"/"retrieve top 5" for both models) -- regenerating
# the manuscript tables now would show new data still labelled/highlighted
# against the OLD winner. Per explicit instruction: stop after the rerun,
# report the new MeanQ decision table below, and let the pins be updated
# by hand before any table regeneration or manuscript recompile.

echo ""
echo "=== new MeanQ decision (compare against current FORCED_REFERENCES pins before updating) ==="
grep -A 30 "Best RAG config by MeanQ" reports/metrics/eu_dev_ablation_results.md || true

echo ""
echo "EU staged rerun orchestrator finished at $(date) -- STOPPED before table regeneration, per instruction."
