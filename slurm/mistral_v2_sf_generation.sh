#!/bin/bash
#SBATCH --job-name=mistral-v2-sf-gen
#SBATCH --array=0-32%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_v2_sf_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_v2_sf_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

CONFIGS=(
  /home/igutierrez134/med_rag_thesis/configs/experiments/963_mistral7b_no_rag_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/964_mistral7b_rag_e5_topk1_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/965_mistral7b_rag_e5_topk3_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/966_mistral7b_rag_e5_topk5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/967_mistral7b_rag_e5_rerank1_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/968_mistral7b_rag_e5_rerank3_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/969_mistral7b_rag_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/970_mistral7b_3shot_no_rag_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/971_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/972_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/973_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/974_mistral7b_no_rag_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/975_mistral7b_rag_e5_topk1_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/976_mistral7b_rag_e5_topk3_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/977_mistral7b_rag_e5_topk5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/978_mistral7b_rag_e5_rerank1_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/979_mistral7b_rag_e5_rerank3_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/980_mistral7b_rag_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/981_mistral7b_3shot_no_rag_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/982_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/983_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/984_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/985_mistral7b_no_rag_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/986_mistral7b_rag_e5_topk1_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/987_mistral7b_rag_e5_topk3_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/988_mistral7b_rag_e5_topk5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/989_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/990_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/991_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/992_mistral7b_3shot_no_rag_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/993_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/994_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/995_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_sf_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "mistral-v2-sf-gen started on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}"
nvidia-smi || true

python scripts/run_generation_from_config.py --config "$CONFIG" --runs 1

echo "mistral-v2-sf-gen finished at $(date)"
