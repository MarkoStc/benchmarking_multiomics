#!/bin/bash -l
#SBATCH --job-name=harvest
#SBATCH --partition=academic
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=7000
#SBATCH --output=report/logs/harvest-%A_%a.out
#SBATCH --error=report/logs/harvest-%A_%a.err

# Stage-1 harvest: one array task per (model, dataset, axis) from report/manifest.txt.
# Reconstructs held-out predictions from saved fold_models and writes the canonical
# metric/prediction shards. mofa uses the mofa conda env; all others use ml.
set -u
ROOT=/work/scitas-share/FellayMultiOmic/code/full-test-pipeline
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ROOT/report/manifest.txt")
read -r MODEL DATASET AXIS <<< "$LINE"
echo "$(date) task ${SLURM_ARRAY_TASK_ID}: $MODEL $DATASET $AXIS"

source ~/miniconda3/etc/profile.d/conda.sh
if [ "$MODEL" = "mofa" ]; then
  conda activate mofa
else
  conda activate ml
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$ROOT/report"
python -u harvest.py "$MODEL" "$DATASET" "$AXIS"
