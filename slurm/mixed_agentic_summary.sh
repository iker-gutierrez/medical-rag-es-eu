#!/bin/bash
#SBATCH --job-name=medrag-mixed-agentic-summary
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mixed_agentic_summary_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/mixed_agentic_summary_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

python scripts/write_mixed_agentic_summary.py

echo "Mixed agentic summary finished at $(date)"
