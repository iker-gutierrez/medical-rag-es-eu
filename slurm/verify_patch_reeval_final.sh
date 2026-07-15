#!/bin/bash
#SBATCH --job-name=final-patch-eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/final_patch_eval_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/final_patch_eval_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "final-patch-eval started on $(hostname) at $(date)"

echo ""
echo "=== Verifying round-3 repetition fix (sns1064_00200 on Latxa) ==="
set +e
python scripts/verify_rerun_truncation.py \
  --manifest /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_round3.json \
  --config-key round3_config \
  --output-key round3_output
VERIFY_EXIT=$?
set -e

if [ $VERIFY_EXIT -ne 0 ]; then
  echo ""
  echo "Round-3 (repetition-penalty) verification still reports issues (exit $VERIFY_EXIT)."
  echo "The presence_penalty=0.3 bump did not resolve the repetition loop for sns1064_00200."
  echo "This record will be left out of the patch; inspect manually before proceeding further."
fi

echo ""
echo "=== Patching round 1 (82 clean) + round 2 (5 clean, sns1064_00200 excluded) ==="
python scripts/patch_truncated_predictions.py \
  --manifest /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest.json \
  --output-key rerun_output

python scripts/patch_truncated_predictions.py \
  --manifest /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_round2.json \
  --output-key round2_output

if [ $VERIFY_EXIT -eq 0 ]; then
  echo ""
  echo "=== Patching round 3 (sns1064_00200 repetition fix) ==="
  python scripts/patch_truncated_predictions.py \
    --manifest /tmp/claude-1034/-home-igutierrez134/822972dc-80bc-4f28-8df8-cdd479f4aca8/scratchpad/truncation_reruns/rerun_manifest_round3.json \
    --output-key round3_output
else
  echo ""
  echo "Skipping round-3 patch (still truncated/looping); sns1064_00200 remains as originally generated for now."
fi

echo ""
echo "=== Re-evaluating all patched runs ==="
python scripts/reeval_patched_runs.py

echo ""
echo "=== Regenerating ES ablation report ==="
python scripts/write_mixed_es_seed_summary.py

echo ""
echo "final-patch-eval finished at $(date)"
