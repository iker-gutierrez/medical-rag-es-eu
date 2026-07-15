#!/bin/bash
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

dep_args=()
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  dep_args=(--dependency=afterok:"$1")
fi

nosf_job="$(sbatch --parsable "${dep_args[@]}" slurm/mistral_v2_generation.sh)"
echo "Submitted Mistral-7B v2 noSF generation: ${nosf_job}"

sf_job="$(sbatch --parsable --dependency=afterok:${nosf_job} slurm/mistral_v2_sf_generation.sh)"
echo "Submitted Mistral-7B v2 SF generation (after noSF): ${sf_job}"

eval_job="$(sbatch --parsable --dependency=afterok:${sf_job} slurm/mistral_v2_evaluation.sh)"
echo "Submitted Mistral-7B v2 evaluation (after SF): ${eval_job}"
