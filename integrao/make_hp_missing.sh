#!/bin/bash
# Generate experiments_integrao_missing.txt
# Each line = one SLURM array task (fixed HP combo × one missing fraction)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-/work/scitas-share/FellayMultiOmic/svm_logreg/brca_real_multiomics.joblib}"
OMICS="${2:-mrna,dnam,rppa,mirna,cnv}"
OUTROOT="${3:-${SCRIPT_DIR}/results_integrao_missing}"

: > "${SCRIPT_DIR}/experiments_integrao_missing.txt"

for mode in supervised unsupervised; do
  for k in 50 100 200; do
    for nb in 10 20; do
      for emb in 32 64; do
        for frac in 0.0 0.25 0.5 0.75 1.0; do
          if [ "$mode" = "unsupervised" ]; then
            for lrC in 0.1 1.0 10.0; do
              echo "python ${SCRIPT_DIR}/run_missing.py \
--dataset-path ${DATASET} \
--omics ${OMICS} \
--mode ${mode} \
--k-per-omic ${k} \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--align-epochs 1000 \
--lr-C ${lrC} \
--include-non-intersection-frac ${frac} \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_missing.txt"
            done
          else
            echo "python ${SCRIPT_DIR}/run_missing.py \
--dataset-path ${DATASET} \
--omics ${OMICS} \
--mode ${mode} \
--k-per-omic ${k} \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--finetune-epochs 1000 \
--align-epochs 1000 \
--include-non-intersection-frac ${frac} \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_missing.txt"
          fi
        done
      done
    done
  done
done

N=$(wc -l < "${SCRIPT_DIR}/experiments_integrao_missing.txt")
echo "Generated ${N} jobs -> experiments_integrao_missing.txt"
echo "Submit with:"
echo "  sbatch --array=1-${N} ${SCRIPT_DIR}/slurm_integrao_array.sh experiments_integrao_missing.txt"
