#!/bin/bash
# Chain: Mistral -> Qwen3.5-9B -> Llama -> Latxa
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

mistral=$(sbatch --parsable slurm/mistral_mixed_nosf_generation.sh)
echo "Mistral:     ${mistral}"

qwen=$(sbatch --parsable --dependency=afterok:${mistral} slurm/qwen35_9b_no_think_generation.sh)
echo "Qwen3.5-9B:  ${qwen} (after ${mistral})"

llama=$(sbatch --parsable --dependency=afterok:${qwen} slurm/llama31_mixed_eu_generation.sh)
echo "Llama:       ${llama} (after ${qwen})"

latxa=$(sbatch --parsable --dependency=afterok:${llama} slurm/latxa_mixed_eu_generation.sh)
echo "Latxa:       ${latxa} (after ${llama})"
