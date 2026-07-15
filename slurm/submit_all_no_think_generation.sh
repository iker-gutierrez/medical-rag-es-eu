#!/bin/bash
# Submit mixed-only no-think generation jobs for Spanish models.
# Each array job runs 3 seeded runs automatically (num_runs=3 in configs).
# Llama/Latxa are Basque-only and are not included here.
set -euo pipefail

cd /home/igutierrez134/med_rag_thesis

echo "Submitting no-think generation jobs (mixed Spanish dataset)..."

q35_9b_id=$(sbatch --parsable slurm/qwen35_9b_no_think_generation.sh)
echo "  Qwen3.5-9B no-think: ${q35_9b_id}"

q35_4b_id=$(sbatch --parsable slurm/qwen35_4b_no_think_generation.sh)
echo "  Qwen3.5-4B no-think: ${q35_4b_id}"

q3_8b_id=$(sbatch --parsable slurm/qwen3_8b_no_think_generation.sh)
echo "  Qwen3-8B no-think: ${q3_8b_id}"

mistral_id=$(sbatch --parsable slurm/mistral_no_think_generation.sh)
echo "  Mistral-7B no-think: ${mistral_id}"

echo ""
echo "All submitted. Job IDs: ${q35_9b_id} ${q35_4b_id} ${q3_8b_id} ${mistral_id}"
