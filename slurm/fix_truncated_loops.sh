#!/bin/bash
#SBATCH --job-name=fix-loops
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=8GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/fix_loops_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/fix_loops_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# CPU-only, no GPU needed -- runs the post-hoc loop-fix on one family's
# completed _v2 output right after that family's generation job finishes
# (submitted with --dependency=afterok:<family job id>, which slurm
# resolves as "wait for every array task of that job to succeed" for an
# array job). Doesn't gate the next family's generation start since it
# uses no GPU -- the generation chain and these fix jobs run independently
# once each is submitted.
#
# ID_MIN/ID_MAX select the family by its config-id range (see
# MODEL_FAMILY_IDS in run_casionly_regeneration.py). NAME_CONTAINS/
# NAME_EXCLUDES resolve the one real ambiguity in this codebase: id 1270
# is used by BOTH a Mistral config and a Qwen-think config, and
# "no_think" contains "think" as a substring so a plain name match isn't
# enough to separate qwen-nothink from qwen-think either.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate

: "${ID_MIN:?Set ID_MIN via sbatch --export}"
: "${ID_MAX:?Set ID_MAX via sbatch --export}"
NAME_CONTAINS="${NAME_CONTAINS:-}"
NAME_EXCLUDES="${NAME_EXCLUDES:-}"

ARGS=(--run-dir-glob "*_v2_seed*" --config-id-min "$ID_MIN" --config-id-max "$ID_MAX")
if [ -n "$NAME_CONTAINS" ]; then
  ARGS+=(--name-contains "$NAME_CONTAINS")
fi
if [ -n "$NAME_EXCLUDES" ]; then
  ARGS+=(--name-excludes "$NAME_EXCLUDES")
fi

echo "fix-truncated-loops id_range=${ID_MIN}-${ID_MAX} name_contains=${NAME_CONTAINS:-<none>} name_excludes=${NAME_EXCLUDES:-<none>} started at $(date)"
python scripts/fix_truncated_loops_in_place.py "${ARGS[@]}"
echo "fix-truncated-loops id_range=${ID_MIN}-${ID_MAX} finished at $(date)"
