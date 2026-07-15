#!/bin/bash
# Submit a full Mistral->Qwen->Llama->Latxa chain with a given seed override.
# Usage: bash submit_all_mixed_seed.sh <seed> [after_jobid]
set -euo pipefail

SEED=${1:?Usage: $0 <seed> [after_jobid]}
AFTER=${2:-}

cd /home/igutierrez134/med_rag_thesis

dep_arg=""
[ -n "$AFTER" ] && dep_arg="--dependency=afterok:${AFTER}"

mistral=$(sbatch --parsable --export=ALL,SEED=${SEED} $dep_arg slurm/mistral_mixed_nosf_generation.sh)
echo "Mistral (seed=${SEED}):     ${mistral}"

qwen=$(sbatch --parsable --export=ALL,SEED=${SEED} --dependency=afterok:${mistral} slurm/qwen35_9b_no_think_generation.sh)
echo "Qwen3.5-9B (seed=${SEED}):  ${qwen}"

llama=$(sbatch --parsable --export=ALL,SEED=${SEED} --dependency=afterok:${qwen} slurm/llama31_mixed_eu_generation.sh)
echo "Llama (seed=${SEED}):       ${llama}"

latxa=$(sbatch --parsable --export=ALL,SEED=${SEED} --dependency=afterok:${llama} slurm/latxa_mixed_eu_generation.sh)
echo "Latxa (seed=${SEED}):       ${latxa}"

echo "${latxa}"
