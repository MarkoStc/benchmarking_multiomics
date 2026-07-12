#!/bin/bash -l
#SBATCH --job-name=integrao-rerun
#SBATCH --partition=academic
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=7000
#SBATCH --output=logs/array-%A_%a.out
#SBATCH --error=logs/array-%A_%a.err

# Usage:
#   sbatch --array=1-N slurm_integrao_array.sh <experiments_file.txt>
#
# IntegrAO uses GPU when available; request one with:
#   sbatch --array=1-N --gres=gpu:1 slurm_integrao_array.sh <experiments_file.txt>

set -euo pipefail

cd /work/scitas-share/FellayMultiOmic/code/full-test-pipeline/integrao
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml  # must have integrao, torch, torch-geometric installed

EXPERIMENT_FILE="${1:-experiments_integrao_missing.txt}"
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${EXPERIMENT_FILE}")

echo "=================================================="
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node:          ${SLURMD_NODENAME:-unknown}"
echo "Experiment:    ${EXPERIMENT_FILE}"
echo "Command:       ${CMD}"
echo "=================================================="

eval "${CMD}"
