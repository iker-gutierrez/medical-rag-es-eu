#!/bin/bash
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

gen_job="$(sbatch --parsable slurm/qwen35_4b_full_spanish_generation.sh)"
echo "Submitted Qwen3.5-4B Spanish noSF generation: ${gen_job}"

sf_job="$(sbatch --parsable --dependency=afterok:${gen_job} slurm/qwen35_4b_full_spanish_sf_generation.sh)"
echo "Submitted Qwen3.5-4B Spanish SF generation: ${sf_job}"

eval_job="$(sbatch --parsable --dependency=afterok:${sf_job} slurm/qwen35_4b_full_spanish_evaluation.sh)"
echo "Submitted Qwen3.5-4B Spanish evaluation: ${eval_job}"
