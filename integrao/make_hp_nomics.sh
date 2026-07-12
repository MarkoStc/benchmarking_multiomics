#!/bin/bash
# Generate experiments_integrao_nomics.txt
# Tests all omic subsets of size 2..5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="${1:-/work/scitas-share/FellayMultiOmic/svm_logreg/brca_real_multiomics.joblib}"
OUTROOT="${2:-${SCRIPT_DIR}/results_integrao_nomics}"

: > "${SCRIPT_DIR}/experiments_integrao_nomics.txt"

# All subsets of size 2-5 from the 5 omics
ALL_OMICS=(mrna dnam rppa mirna cnv)
COMBOS=(
  "mrna,dnam"
  "mrna,rppa"
  "mrna,mirna"
  "mrna,cnv"
  "dnam,rppa"
  "dnam,mirna"
  "dnam,cnv"
  "rppa,mirna"
  "rppa,cnv"
  "mirna,cnv"
  "mrna,dnam,rppa"
  "mrna,dnam,mirna"
  "mrna,dnam,cnv"
  "mrna,rppa,mirna"
  "mrna,rppa,cnv"
  "mrna,mirna,cnv"
  "dnam,rppa,mirna"
  "dnam,rppa,cnv"
  "dnam,mirna,cnv"
  "rppa,mirna,cnv"
  "mrna,dnam,rppa,mirna"
  "mrna,dnam,rppa,cnv"
  "mrna,dnam,mirna,cnv"
  "mrna,rppa,mirna,cnv"
  "dnam,rppa,mirna,cnv"
  "mrna,dnam,rppa,mirna,cnv"
)

for mode in supervised unsupervised; do
  for combo in "${COMBOS[@]}"; do
    for nb in 10 20; do
      for emb in 32 64; do
        if [ "$mode" = "unsupervised" ]; then
          echo "python ${SCRIPT_DIR}/run_nomics.py \
--dataset-path ${DATASET} \
--omics ${combo} \
--mode ${mode} \
--k-per-omic 100 \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--align-epochs 1000 \
--lr-C 1.0 \
--include-non-intersection-frac 1.0 \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_nomics.txt"
        else
          echo "python ${SCRIPT_DIR}/run_nomics.py \
--dataset-path ${DATASET} \
--omics ${combo} \
--mode ${mode} \
--k-per-omic 100 \
--neighbor-size ${nb} \
--embedding-dims ${emb} \
--finetune-epochs 1000 \
--align-epochs 1000 \
--include-non-intersection-frac 1.0 \
--outer-splits 5 \
--output-root ${OUTROOT}" >> "${SCRIPT_DIR}/experiments_integrao_nomics.txt"
        fi
      done
    done
  done
done

N=$(wc -l < "${SCRIPT_DIR}/experiments_integrao_nomics.txt")
echo "Generated ${N} jobs -> experiments_integrao_nomics.txt"
echo "Submit with:"
echo "  sbatch --array=1-${N} ${SCRIPT_DIR}/slurm_integrao_array.sh experiments_integrao_nomics.txt"
