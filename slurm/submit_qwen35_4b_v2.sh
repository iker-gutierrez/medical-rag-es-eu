#!/bin/bash
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

nosf_job="$(sbatch --parsable slurm/qwen35_4b_v2_generation.sh)"
echo "Submitted qwen35_4b_v2 noSF generation: ${nosf_job}"

sf_job="$(sbatch --parsable --dependency=afterok:${nosf_job} slurm/qwen35_4b_v2_sf_generation.sh)"
echo "Submitted qwen35_4b_v2 SF generation (after noSF): ${sf_job}"

eval_job="$(sbatch --parsable --dependency=afterok:${sf_job} slurm/qwen35_4b_v2_evaluation.sh)"
echo "Submitted qwen35_4b_v2 evaluation (after SF): ${eval_job}"
