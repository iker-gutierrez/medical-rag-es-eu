#!/bin/bash
# Submit 2 parallel generation chains + evaluation.
# Job 6195 (exp 17) must already be running before calling this.
# Run from project root: bash slurm/submit_parallel_rerun.sh

set -euo pipefail

LOGDIR=experiments/slurm_logs

# Chain A: exps 20, 29, 31, 33, 37
CHAIN_A=$(sbatch slurm/gen_chain_a.sh | awk '{print $NF}')
echo "Chain A (20,29,31,33,37) → job array ${CHAIN_A}"

# Chain B: exps 28, 30, 32, 36, 38  (starts immediately on second GPU)
CHAIN_B=$(sbatch slurm/gen_chain_b.sh | awk '{print $NF}')
echo "Chain B (28,30,32,36,38) → job array ${CHAIN_B}"

# Evaluation after exp 17 (6195) and both chains complete
EVAL_JOB=$(sbatch \
  --dependency="afterok:6195,afterok:${CHAIN_A},afterok:${CHAIN_B}" \
  slurm/full_rerun_evaluation.sh | awk '{print $NF}')
echo "Evaluation → job ${EVAL_JOB} (after 6195, ${CHAIN_A}, ${CHAIN_B})"
