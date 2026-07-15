#!/bin/bash
#SBATCH --job-name=mistral-v2-gen
#SBATCH --array=0-32%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_v2_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_v2_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

CONFIGS=(
  /home/igutierrez134/med_rag_thesis/configs/experiments/864_mistral7b_no_rag_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/865_mistral7b_rag_e5_topk1_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/866_mistral7b_rag_e5_topk3_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/867_mistral7b_rag_e5_topk5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/868_mistral7b_rag_e5_rerank1_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/869_mistral7b_rag_e5_rerank3_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/870_mistral7b_rag_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/871_mistral7b_3shot_no_rag_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/872_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/873_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/874_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_sns1064_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/875_mistral7b_no_rag_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/876_mistral7b_rag_e5_topk1_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/877_mistral7b_rag_e5_topk3_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/878_mistral7b_rag_e5_topk5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/879_mistral7b_rag_e5_rerank1_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/880_mistral7b_rag_e5_rerank3_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/881_mistral7b_rag_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/882_mistral7b_3shot_no_rag_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/883_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/884_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/885_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/886_mistral7b_no_rag_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/887_mistral7b_rag_e5_topk1_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/888_mistral7b_rag_e5_topk3_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/889_mistral7b_rag_e5_topk5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/890_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/891_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/892_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/893_mistral7b_3shot_no_rag_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/894_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/895_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_dev.json
  /home/igutierrez134/med_rag_thesis/configs/experiments/896_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_dev.json
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "mistral-v2-gen started on $(hostname) at $(date)"
echo "CONFIG=${CONFIG}"
nvidia-smi || true

python scripts/run_generation_from_config.py --config "$CONFIG" --runs 1

echo "mistral-v2-gen finished at $(date)"
