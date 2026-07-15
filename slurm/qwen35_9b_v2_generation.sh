#!/bin/bash
#SBATCH --job-name=qwen35_9b_v2-gen
#SBATCH --array=0-32%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_v2_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_v2_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

CONFIGS=(
  /home/igutierrez134/med_rag_thesis/configs/experiments/600_qwen35_9b_no_rag_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/601_qwen35_9b_rag_e5_topk1_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/602_qwen35_9b_rag_e5_topk3_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/603_qwen35_9b_rag_e5_topk5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/604_qwen35_9b_rag_e5_rerank1_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/605_qwen35_9b_rag_e5_rerank3_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/606_qwen35_9b_rag_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/607_qwen35_9b_3shot_no_rag_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/608_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/609_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/610_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/622_qwen35_9b_no_rag_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/623_qwen35_9b_rag_e5_topk1_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/624_qwen35_9b_rag_e5_topk3_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/625_qwen35_9b_rag_e5_topk5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/626_qwen35_9b_rag_e5_rerank1_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/627_qwen35_9b_rag_e5_rerank3_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/628_qwen35_9b_rag_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/629_qwen35_9b_3shot_no_rag_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/630_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/631_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/632_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/644_qwen35_9b_no_rag_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/645_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/646_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/647_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/648_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/649_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/650_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/651_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/652_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/653_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/654_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "qwen35_9b_v2-gen started on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}"
nvidia-smi || true

python scripts/run_generation_from_config.py --config "$CONFIG" --runs 1

echo "qwen35_9b_v2-gen finished at $(date)"
