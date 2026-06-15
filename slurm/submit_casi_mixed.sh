#!/bin/bash
# Submit CasiMedicos-dev and mixed-dev generation chains in parallel,
# then evaluation + summary update.
# Run from project root: bash slurm/submit_casi_mixed.sh

set -euo pipefail

# CasiMedicos dev chain (exps 39-49) — uses GPU 1
CHAIN_CASI=$(sbatch slurm/gen_chain_casimedicos.sh | awk '{print $NF}')
echo "CasiMedicos-dev chain (39-49) → job array ${CHAIN_CASI}"

# Mixed dev chain (exps 50-60) — uses GPU 2 simultaneously
CHAIN_MIXED=$(sbatch slurm/gen_chain_mixed.sh | awk '{print $NF}')
echo "Mixed-dev chain (50-60)       → job array ${CHAIN_MIXED}"

# Evaluation + summary update after both chains complete
EVAL_JOB=$(sbatch \
  --dependency="afterok:${CHAIN_CASI},afterok:${CHAIN_MIXED}" \
  slurm/eval_casi_mixed.sh | awk '{print $NF}')
echo "Evaluation + summary update   → job ${EVAL_JOB} (after ${CHAIN_CASI}, ${CHAIN_MIXED})"
