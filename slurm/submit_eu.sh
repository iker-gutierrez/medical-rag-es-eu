#!/bin/bash
# Submit the full Basque pipeline with strict 1-GPU-at-a-time gen chains.
# Run from project root: bash slurm/submit_eu.sh [WAIT_JOB_ID]
#
# Optional: pass a job ID to wait for before starting (e.g. the overnight eval job).
#   bash slurm/submit_eu.sh 6220
#
# GPU budget: at most 1 GPU used by the EU pipeline at any time, so the second
# GPU remains free for any other job (Codex Agentic Reasoner, etc.).
#
# Pipeline (strictly sequential GPU use):
#   translate_eu (1 GPU)
#     → index_eu (1 GPU)
#       → gen_chain_sns1064_eu (1 GPU, 11 tasks)
#         → gen_chain_casimedicos_eu (1 GPU, 11 tasks)
#           → gen_chain_mixed_eu (1 GPU, 11 tasks)
#             → eval_eu (CPU)

set -euo pipefail

WAIT_JOB="${1:-}"

# Step 1: translate (1 GPU)
if [[ -n "$WAIT_JOB" ]]; then
  TRANSLATE=$(sbatch --dependency="afterok:${WAIT_JOB}" slurm/translate_eu.sh | awk '{print $NF}')
else
  TRANSLATE=$(sbatch slurm/translate_eu.sh | awk '{print $NF}')
fi
echo "Translate EU     → job ${TRANSLATE}"

# Step 2: index (1 GPU, after translate)
INDEX=$(sbatch --dependency="afterok:${TRANSLATE}" slurm/index_eu.sh | awk '{print $NF}')
echo "Index EU         → job ${INDEX} (after ${TRANSLATE})"

# Step 3: sns1064_eu and casimedicos_eu run in parallel (1 GPU each = 2 GPUs total)
SNS_EU=$(sbatch --dependency="afterok:${INDEX}" slurm/gen_chain_sns1064_eu.sh | awk '{print $NF}')
echo "Gen SNS EU       → job array ${SNS_EU} (after ${INDEX})"

CASI_EU=$(sbatch --dependency="afterok:${INDEX}" slurm/gen_chain_casimedicos_eu.sh | awk '{print $NF}')
echo "Gen CasiMedicos EU → job array ${CASI_EU} (after ${INDEX}, parallel with SNS)"

# Step 4: mixed_eu starts only after both parallel chains finish
MIXED_EU=$(sbatch --dependency="afterok:${SNS_EU}:${CASI_EU}" slurm/gen_chain_mixed_eu.sh | awk '{print $NF}')
echo "Gen Mixed EU     → job array ${MIXED_EU} (after ${CASI_EU})"

# Step 4: evaluate (CPU, after all gen chains)
EVAL=$(sbatch --dependency="afterok:${MIXED_EU}" slurm/eval_eu.sh | awk '{print $NF}')
echo "Eval EU          → job ${EVAL} (after ${MIXED_EU})"

echo ""
echo "Full Basque pipeline submitted (2 GPUs: sns+casi parallel, then mixed):"
echo "  translate=${TRANSLATE}  index=${INDEX}"
echo "  sns_eu=${SNS_EU}  casi_eu=${CASI_EU} (parallel)"
echo "  mixed_eu=${MIXED_EU}  eval=${EVAL}"
