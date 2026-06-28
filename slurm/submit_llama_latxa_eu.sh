#!/bin/bash
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

generation_job_id="$(sbatch --parsable slurm/llama_latxa_eu_generation.sh)"
echo "Submitted Llama/Latxa EU generation array: ${generation_job_id}"

evaluation_job_id="$(sbatch --parsable --dependency=afterok:${generation_job_id} slurm/llama_latxa_eu_evaluation.sh)"
echo "Submitted dependent Llama/Latxa EU evaluation: ${evaluation_job_id}"
