#!/bin/bash
# Chain: Llama (noSF+SF) -> Latxa (noSF+SF)
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

llama=$(sbatch --parsable slurm/llama31_mixed_eu_generation.sh)
echo "Llama: ${llama}"

latxa=$(sbatch --parsable --dependency=afterok:${llama} slurm/latxa_mixed_eu_generation.sh)
echo "Latxa: ${latxa} (after ${llama})"
