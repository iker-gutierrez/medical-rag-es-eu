#!/bin/bash
#SBATCH --job-name=latxa-backfill
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/latxa_backfill_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/latxa_backfill_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# Backfill the 3 Latxa v2 configs (1056, 1058, 1060) x 3 seeds that were
# never generated -- their shard (9032_1) was cancelled early to free a
# GPU for prompt-variant testing, and only got through 1052/1054 before
# cancellation. This uses the SAME real config files and generation
# mechanism as the sharded chain, just invoked directly rather than
# through the shard-splitting driver, since only 3 of 11 configs are
# missing (not a fair 50/50 split worth re-sharding for).
for cfg_id in 1056 1058 1060; do
  cfg_path=$(ls configs/experiments/${cfg_id}_latxa*_v2.json)
  for seed in 42 43 44; do
    echo ""
    echo "=== $cfg_path seed=$seed ==="
    python scripts/run_generation_from_config.py --config "$cfg_path" --seed "$seed"
  done
done

echo ""
echo "Latxa backfill complete."
