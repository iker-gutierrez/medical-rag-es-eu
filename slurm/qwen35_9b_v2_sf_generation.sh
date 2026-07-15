#!/bin/bash
#SBATCH --job-name=qwen35_9b_v2_sf-gen
#SBATCH --array=0-32%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_v2_sf_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_v2_sf_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

CONFIGS=(
  /home/igutierrez134/med_rag_thesis/configs/experiments/666_qwen35_9b_no_rag_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/667_qwen35_9b_rag_e5_topk1_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/668_qwen35_9b_rag_e5_topk3_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/669_qwen35_9b_rag_e5_topk5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/670_qwen35_9b_rag_e5_rerank1_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/671_qwen35_9b_rag_e5_rerank3_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/672_qwen35_9b_rag_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/673_qwen35_9b_3shot_no_rag_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/674_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/675_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/676_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_sns1064_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/688_qwen35_9b_no_rag_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/689_qwen35_9b_rag_e5_topk1_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/690_qwen35_9b_rag_e5_topk3_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/691_qwen35_9b_rag_e5_topk5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/692_qwen35_9b_rag_e5_rerank1_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/693_qwen35_9b_rag_e5_rerank3_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/694_qwen35_9b_rag_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/695_qwen35_9b_3shot_no_rag_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/696_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/697_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/698_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/710_qwen35_9b_no_rag_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/711_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/712_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/713_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/714_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/715_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/716_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/717_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/718_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/719_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_sf_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/720_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_sf_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "qwen35_9b_v2_sf-gen started on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}"
nvidia-smi || true

python scripts/run_generation_from_config.py --config "$CONFIG" --runs 1

echo "qwen35_9b_v2_sf-gen finished at $(date)"
