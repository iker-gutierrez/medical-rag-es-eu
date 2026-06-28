#!/bin/bash
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

dependency_args=()
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  dependency_args=(--dependency=afterany:"$1")
fi

generation_job_id="$(sbatch --parsable "${dependency_args[@]}" slurm/mistral_spanish_generation.sh)"
echo "Submitted Mistral Spanish generation array: ${generation_job_id}"

evaluation_job_id="$(sbatch --parsable --dependency=afterok:"${generation_job_id}" slurm/mistral_spanish_evaluation.sh)"
echo "Submitted dependent Mistral Spanish evaluation: ${evaluation_job_id}"
