#!/bin/bash
#SBATCH --job-name=mistral-v2-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_v2_evaluation_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mistral_v2_evaluation_%j.err
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
  864_mistral7b_no_rag_no_think_extractive_sns1064
  865_mistral7b_rag_e5_topk1_no_think_extractive_sns1064
  866_mistral7b_rag_e5_topk3_no_think_extractive_sns1064
  867_mistral7b_rag_e5_topk5_no_think_extractive_sns1064
  868_mistral7b_rag_e5_rerank1_no_think_extractive_sns1064
  869_mistral7b_rag_e5_rerank3_no_think_extractive_sns1064
  870_mistral7b_rag_e5_rerank5_no_think_extractive_sns1064
  871_mistral7b_3shot_no_rag_no_think_extractive_sns1064
  872_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064
  873_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_sns1064
  874_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_sns1064
  875_mistral7b_no_rag_no_think_extractive_casimedicos
  876_mistral7b_rag_e5_topk1_no_think_extractive_casimedicos
  877_mistral7b_rag_e5_topk3_no_think_extractive_casimedicos
  878_mistral7b_rag_e5_topk5_no_think_extractive_casimedicos
  879_mistral7b_rag_e5_rerank1_no_think_extractive_casimedicos
  880_mistral7b_rag_e5_rerank3_no_think_extractive_casimedicos
  881_mistral7b_rag_e5_rerank5_no_think_extractive_casimedicos
  882_mistral7b_3shot_no_rag_no_think_extractive_casimedicos
  883_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos
  884_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos
  885_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos
  886_mistral7b_no_rag_no_think_extractive_mixed
  887_mistral7b_rag_e5_topk1_no_think_extractive_mixed
  888_mistral7b_rag_e5_topk3_no_think_extractive_mixed
  889_mistral7b_rag_e5_topk5_no_think_extractive_mixed
  890_mistral7b_rag_e5_rerank1_no_think_extractive_mixed
  891_mistral7b_rag_e5_rerank3_no_think_extractive_mixed
  892_mistral7b_rag_e5_rerank5_no_think_extractive_mixed
  893_mistral7b_3shot_no_rag_no_think_extractive_mixed
  894_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed
  895_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed
  896_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed
  963_mistral7b_no_rag_no_think_extractive_sns1064_sf
  964_mistral7b_rag_e5_topk1_no_think_extractive_sns1064_sf
  965_mistral7b_rag_e5_topk3_no_think_extractive_sns1064_sf
  966_mistral7b_rag_e5_topk5_no_think_extractive_sns1064_sf
  967_mistral7b_rag_e5_rerank1_no_think_extractive_sns1064_sf
  968_mistral7b_rag_e5_rerank3_no_think_extractive_sns1064_sf
  969_mistral7b_rag_e5_rerank5_no_think_extractive_sns1064_sf
  970_mistral7b_3shot_no_rag_no_think_extractive_sns1064_sf
  971_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_sns1064_sf
  972_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_sns1064_sf
  973_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_sns1064_sf
  974_mistral7b_no_rag_no_think_extractive_casimedicos_sf
  975_mistral7b_rag_e5_topk1_no_think_extractive_casimedicos_sf
  976_mistral7b_rag_e5_topk3_no_think_extractive_casimedicos_sf
  977_mistral7b_rag_e5_topk5_no_think_extractive_casimedicos_sf
  978_mistral7b_rag_e5_rerank1_no_think_extractive_casimedicos_sf
  979_mistral7b_rag_e5_rerank3_no_think_extractive_casimedicos_sf
  980_mistral7b_rag_e5_rerank5_no_think_extractive_casimedicos_sf
  981_mistral7b_3shot_no_rag_no_think_extractive_casimedicos_sf
  982_mistral7b_rag_cross_domain_e5_rerank5_no_think_extractive_casimedicos_sf
  983_mistral7b_rag_mixed_e5_rerank5_no_think_extractive_casimedicos_sf
  984_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_casimedicos_sf
  985_mistral7b_no_rag_no_think_extractive_mixed_sf
  986_mistral7b_rag_e5_topk1_no_think_extractive_mixed_sf
  987_mistral7b_rag_e5_topk3_no_think_extractive_mixed_sf
  988_mistral7b_rag_e5_topk5_no_think_extractive_mixed_sf
  989_mistral7b_rag_e5_rerank1_no_think_extractive_mixed_sf
  990_mistral7b_rag_e5_rerank3_no_think_extractive_mixed_sf
  991_mistral7b_rag_e5_rerank5_no_think_extractive_mixed_sf
  992_mistral7b_3shot_no_rag_no_think_extractive_mixed_sf
  993_mistral7b_rag_sns1064_e5_rerank5_no_think_extractive_mixed_sf
  994_mistral7b_rag_casimedicos_e5_rerank5_no_think_extractive_mixed_sf
  995_mistral7b_rag_3shot_e5_rerank5_no_think_extractive_mixed_sf
)

echo "mistral-v2-eval evaluation started on $(hostname) at $(date)"

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

echo "mistral-v2-eval evaluation finished at $(date)"
