#!/bin/bash
# Generate pca_logreg experiment files for all datasets and axes.
# HP grid swept across jobs (one fixed combo per job), identical benchmark
# standard to the other models: seeds 0/1/2, missing fracs, feature ratios,
# n-patients, 31 nomics combos, outer/inner CV 5/3.
#   missing | ratio : full HP grid      nomics | npatients : paper defaults
set -euo pipefail

DATA_ROOT="/work/scitas-share/FellayMultiOmic/code/full-test-pipeline/data/processed"
ALL_OMICS="mrna,dnam,rppa,mirna,cnv"

if [[ $# -gt 0 ]]; then DATASETS=("$@"); else DATASETS=("TCGA-BRCA" "TCGA-LGG" "TCGA-KIPAN"); fi

SEEDS=(0 1 2)
FRACS=(0.0 0.25 0.5 0.75 1.0)
RATIOS=(1e-05 0.0001 0.001 0.01 0.02 0.05 0.1 0.2 0.25 0.5 0.75 1.0)
NPATS=(20 50 100 200 300 400 500)
KS=(50 100 200)
K_DEF=100

HP_COMBOS=(
  "--n-components 5 --downstream-c 0.01"
  "--n-components 5 --downstream-c 0.1"
  "--n-components 5 --downstream-c 1.0"
  "--n-components 5 --downstream-c 10.0"
  "--n-components 15 --downstream-c 0.01"
  "--n-components 15 --downstream-c 0.1"
  "--n-components 15 --downstream-c 1.0"
  "--n-components 15 --downstream-c 10.0"
  "--n-components 30 --downstream-c 0.01"
  "--n-components 30 --downstream-c 0.1"
  "--n-components 30 --downstream-c 1.0"
  "--n-components 30 --downstream-c 10.0"
)
HP_DEFAULT="--n-components 15 --downstream-c 1.0"

COMBOS=(
  "cnv" "dnam" "mirna" "mrna" "rppa"
  "mrna,dnam" "mrna,rppa" "mrna,mirna" "mrna,cnv"
  "dnam,rppa" "dnam,mirna" "dnam,cnv"
  "rppa,mirna" "rppa,cnv" "mirna,cnv"
  "mrna,dnam,rppa" "mrna,dnam,mirna" "mrna,dnam,cnv"
  "mrna,rppa,mirna" "mrna,rppa,cnv" "mrna,mirna,cnv"
  "dnam,rppa,mirna" "dnam,rppa,cnv" "dnam,mirna,cnv" "rppa,mirna,cnv"
  "mrna,dnam,rppa,mirna" "mrna,dnam,rppa,cnv" "mrna,dnam,mirna,cnv"
  "mrna,rppa,mirna,cnv" "dnam,rppa,mirna,cnv"
  "mrna,dnam,rppa,mirna,cnv"
)

for DATASET in "${DATASETS[@]}"; do
  DATASET_PATH="${DATA_ROOT}/${DATASET}.joblib"
  COMMON="--dataset-path ${DATASET_PATH} --outer-splits 5"

  OUTFILE="experiments_${DATASET}_missing_hpgrid.txt"; : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do for hp in "${HP_COMBOS[@]}"; do for k in "${KS[@]}"; do for frac in "${FRACS[@]}"; do
    echo "python run_missing.py ${COMMON} --omics ${ALL_OMICS} ${hp} --k-per-omic ${k} --include-non-intersection-frac ${frac} --random-state ${seed} --output-root results/${DATASET}/missing" >> "$OUTFILE"
  done; done; done; done
  echo "${DATASET} missing: $(wc -l < "$OUTFILE") jobs"

  OUTFILE="experiments_${DATASET}_ratio_hpgrid.txt"; : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do for hp in "${HP_COMBOS[@]}"; do for ratio in "${RATIOS[@]}"; do
    echo "python run_ratio.py ${COMMON} --omics ${ALL_OMICS} ${hp} --ratio-per-omic ${ratio} --include-non-intersection-frac 1.0 --random-state ${seed} --output-root results/${DATASET}/ratio" >> "$OUTFILE"
  done; done; done
  echo "${DATASET} ratio: $(wc -l < "$OUTFILE") jobs"

  OUTFILE="experiments_${DATASET}_nomics_defaults.txt"; : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do for combo in "${COMBOS[@]}"; do
    echo "python run_nomics.py ${COMMON} --omics ${combo} ${HP_DEFAULT} --k-per-omic ${K_DEF} --include-non-intersection-frac 1.0 --random-state ${seed} --output-root results/${DATASET}/nomics" >> "$OUTFILE"
  done; done
  echo "${DATASET} nomics: $(wc -l < "$OUTFILE") jobs"

  OUTFILE="experiments_${DATASET}_npatients_defaults.txt"; : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do for n in "${NPATS[@]}"; do
    echo "python run_npatients.py ${COMMON} --omics ${ALL_OMICS} ${HP_DEFAULT} --k-per-omic ${K_DEF} --n-patients ${n} --include-non-intersection-frac 1.0 --random-state ${seed} --output-root results/${DATASET}/npatients" >> "$OUTFILE"
  done; done
  echo "${DATASET} npatients: $(wc -l < "$OUTFILE") jobs"
done
