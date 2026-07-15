#!/bin/bash
#SBATCH --job-name=qwen35_9b_v2-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_v2_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen35_9b_v2_evaluation_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

RUN_IDS=(
  600_qwen35_9b_no_rag_no_think_extractive_sns1064
  601_qwen35_9b_rag_e5_topk1_no_think_extractive_sns1064
  602_qwen35_9b_rag_e5_topk3_no_think_extractive_sns1064
  603_qwen35_9b_rag_e5_topk5_no_think_extractive_sns1064
  604_qwen35_9b_rag_e5_rerank1_no_think_extractive_sns1064
  605_qwen35_9b_rag_e5_rerank3_no_think_extractive_sns1064
  606_qwen35_9b_rag_e5_rerank5_no_think_extractive_sns1064
  607_qwen35_9b_3shot_no_rag_no_think_extractive_sns1064
  608_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064
  609_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_sns1064
  610_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_sns1064
  622_qwen35_9b_no_rag_no_think_extractive_casimedicos
  623_qwen35_9b_rag_e5_topk1_no_think_extractive_casimedicos
  624_qwen35_9b_rag_e5_topk3_no_think_extractive_casimedicos
  625_qwen35_9b_rag_e5_topk5_no_think_extractive_casimedicos
  626_qwen35_9b_rag_e5_rerank1_no_think_extractive_casimedicos
  627_qwen35_9b_rag_e5_rerank3_no_think_extractive_casimedicos
  628_qwen35_9b_rag_e5_rerank5_no_think_extractive_casimedicos
  629_qwen35_9b_3shot_no_rag_no_think_extractive_casimedicos
  630_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos
  631_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos
  632_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos
  644_qwen35_9b_no_rag_no_think_extractive_mixed
  645_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed
  646_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed
  647_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed
  648_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed
  649_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed
  650_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed
  651_qwen35_9b_3shot_no_rag_no_think_extractive_mixed
  652_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed
  653_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed
  654_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed
  666_qwen35_9b_no_rag_no_think_extractive_sns1064_sf
  667_qwen35_9b_rag_e5_topk1_no_think_extractive_sns1064_sf
  668_qwen35_9b_rag_e5_topk3_no_think_extractive_sns1064_sf
  669_qwen35_9b_rag_e5_topk5_no_think_extractive_sns1064_sf
  670_qwen35_9b_rag_e5_rerank1_no_think_extractive_sns1064_sf
  671_qwen35_9b_rag_e5_rerank3_no_think_extractive_sns1064_sf
  672_qwen35_9b_rag_e5_rerank5_no_think_extractive_sns1064_sf
  673_qwen35_9b_3shot_no_rag_no_think_extractive_sns1064_sf
  674_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064_sf
  675_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_sns1064_sf
  676_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_sns1064_sf
  688_qwen35_9b_no_rag_no_think_extractive_casimedicos_sf
  689_qwen35_9b_rag_e5_topk1_no_think_extractive_casimedicos_sf
  690_qwen35_9b_rag_e5_topk3_no_think_extractive_casimedicos_sf
  691_qwen35_9b_rag_e5_topk5_no_think_extractive_casimedicos_sf
  692_qwen35_9b_rag_e5_rerank1_no_think_extractive_casimedicos_sf
  693_qwen35_9b_rag_e5_rerank3_no_think_extractive_casimedicos_sf
  694_qwen35_9b_rag_e5_rerank5_no_think_extractive_casimedicos_sf
  695_qwen35_9b_3shot_no_rag_no_think_extractive_casimedicos_sf
  696_qwen35_9b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos_sf
  697_qwen35_9b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos_sf
  698_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos_sf
  710_qwen35_9b_no_rag_no_think_extractive_mixed_sf
  711_qwen35_9b_rag_e5_topk1_no_think_extractive_mixed_sf
  712_qwen35_9b_rag_e5_topk3_no_think_extractive_mixed_sf
  713_qwen35_9b_rag_e5_topk5_no_think_extractive_mixed_sf
  714_qwen35_9b_rag_e5_rerank1_no_think_extractive_mixed_sf
  715_qwen35_9b_rag_e5_rerank3_no_think_extractive_mixed_sf
  716_qwen35_9b_rag_e5_rerank5_no_think_extractive_mixed_sf
  717_qwen35_9b_3shot_no_rag_no_think_extractive_mixed_sf
  718_qwen35_9b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_sf
  719_qwen35_9b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_sf
  720_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_mixed_sf
)

echo "qwen35_9b_v2-eval evaluation started on $(hostname) at $(date)"

for run_id in "${RUN_IDS[@]}"; do
  echo "=== Evaluating ${run_id} ==="
  pred="experiments/runs/${run_id}/predictions.jsonl"
  out="reports/metrics/${run_id}.json"
  if [[ ! -f "$pred" ]]; then
    echo "WARNING: missing ${pred}, skipping" >&2
    continue
  fi
  python scripts/evaluate_predictions.py \
    --predictions "$pred" \
    --output "$out" \
    --semantic-model intfloat/multilingual-e5-large \
    --bertscore-model bert-base-multilingual-cased \
    --bertscore-lang es
done

echo "qwen35_9b_v2-eval evaluation finished at $(date)"
