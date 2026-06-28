#!/bin/bash
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

generation_job_id="$(sbatch --parsable slurm/qwen3_8b_spanish_generation.sh)"
echo "Submitted Qwen3-8B Spanish generation array: ${generation_job_id}"

evaluation_job_id="$(sbatch --parsable --dependency=afterok:${generation_job_id} slurm/qwen3_8b_spanish_evaluation.sh)"
echo "Submitted dependent Qwen3-8B Spanish evaluation: ${evaluation_job_id}"
