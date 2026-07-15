#!/bin/bash
# Submit Basque EU v2 full pipeline:
#   1. translate_eu   — retranslate all splits (new SNS splits + casimedicos-exp)
#   2. index_eu       — rebuild all three EU retrieval indices from new train sets
#   3. eu-v2 noSF gen — 66 configs (996-1061)
#   4. eu-v2 SF gen   — 66 configs (1062-1127)
#   5. eu-v2 eval     — 132 runs

set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

translate_job="$(sbatch --parsable slurm/translate_eu.sh)"
echo "Submitted EU translation: ${translate_job}"

index_job="$(sbatch --parsable --dependency=afterok:${translate_job} slurm/index_eu.sh)"
echo "Submitted EU indexing (after translation): ${index_job}"

nosf_job="$(sbatch --parsable --dependency=afterok:${index_job} slurm/llama_latxa_eu_v2_generation.sh)"
echo "Submitted EU v2 noSF generation (after indexing): ${nosf_job}"

sf_job="$(sbatch --parsable --dependency=afterok:${nosf_job} slurm/llama_latxa_eu_v2_sf_generation.sh)"
echo "Submitted EU v2 SF generation (after noSF): ${sf_job}"

eval_job="$(sbatch --parsable --dependency=afterok:${sf_job} slurm/llama_latxa_eu_v2_evaluation.sh)"
echo "Submitted EU v2 evaluation (after SF): ${eval_job}"
