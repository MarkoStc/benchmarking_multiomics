#!/bin/bash
# Generate experiments_integrao_ratio.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-/work/scitas-share/FellayMultiOmic/svm_logreg/brca_real_multiomics.joblib}"
OMICS="${2:-mrna,dnam,rppa,mirna,cnv}"
OUTROOT="${3:-${SCRIPT_DIR}/results_integrao_ratio}"

: > "${SCRIPT_DIR}/experiments_integrao_ratio.txt"

for mode in supervised unsupervised; do
  for nb in 10 20; do
    for emb in 32 64; do
      for ratio in 0.01 0.05 0.1 0.2 0.5; do
        if [ "$mode" = "unsupervised" ]; then
          for lrC in 0.1 1.0 10.0; do
            echo "python ${SCRIPT_DIR}/run_ratio.py \
--dataset-path ${DATASET} \
--omics ${OMICS} \
--mode ${mode} \
--ratio-per-omic ${ratio} \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--align-epochs 1000 \
--lr-C ${lrC} \
--include-non-intersection-frac 1.0 \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_ratio.txt"
          done
        else
          echo "python ${SCRIPT_DIR}/run_ratio.py \
--dataset-path ${DATASET} \
--omics ${OMICS} \
--mode ${mode} \
--ratio-per-omic ${ratio} \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--finetune-epochs 1000 \
--align-epochs 1000 \
--include-non-intersection-frac 1.0 \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_ratio.txt"
        fi
      done
    done
  done
done

N=$(wc -l < "${SCRIPT_DIR}/experiments_integrao_ratio.txt")
echo "Generated ${N} jobs -> experiments_integrao_ratio.txt"
echo "Submit with:"
echo "  sbatch --array=1-${N} ${SCRIPT_DIR}/slurm_integrao_array.sh experiments_integrao_ratio.txt"
