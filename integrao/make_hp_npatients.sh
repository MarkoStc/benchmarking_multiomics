#!/bin/bash
# Generate experiments_integrao_npatients.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-/work/scitas-share/FellayMultiOmic/svm_logreg/brca_real_multiomics.joblib}"
OMICS="${2:-mrna,dnam,rppa,mirna,cnv}"
OUTROOT="${3:-${SCRIPT_DIR}/results_integrao_npatients}"

: > "${SCRIPT_DIR}/experiments_integrao_npatients.txt"

for mode in supervised unsupervised; do
  for k in 50 100 200; do
    for nb in 10 20; do
      for emb in 32 64; do
        for n in 100 200 400; do
          if [ "$mode" = "unsupervised" ]; then
            for lrC in 0.1 1.0 10.0; do
              echo "python ${SCRIPT_DIR}/run_npatients.py \
--dataset-path ${DATASET} \
--omics ${OMICS} \
--mode ${mode} \
--k-per-omic ${k} \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--align-epochs 1000 \
--lr-C ${lrC} \
--n-patients ${n} \
--include-non-intersection-frac 1.0 \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_npatients.txt"
            done
          else
            echo "python ${SCRIPT_DIR}/run_npatients.py \
--dataset-path ${DATASET} \
--omics ${OMICS} \
--mode ${mode} \
--k-per-omic ${k} \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--finetune-epochs 1000 \
--align-epochs 1000 \
--n-patients ${n} \
--include-non-intersection-frac 1.0 \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_npatients.txt"
          fi
        done
      done
    done
  done
done

N=$(wc -l < "${SCRIPT_DIR}/experiments_integrao_npatients.txt")
echo "Generated ${N} jobs -> experiments_integrao_npatients.txt"
echo "Submit with:"
echo "  sbatch --array=1-${N} ${SCRIPT_DIR}/slurm_integrao_array.sh experiments_integrao_npatients.txt"
