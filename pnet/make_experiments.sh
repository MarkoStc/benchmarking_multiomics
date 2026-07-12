#!/bin/bash
# Generate P-NET experiment files for all datasets and axes.
#
# HP search grid:
#   hidden_units : 64 128
#   dropout      : 0.1 0.3 0.5
#   w_reg        : 0.001 0.01 0.1
#
# Shared benchmark standard (matches old MOFA design; identical across all models):
#   seeds          : 0 1 2
#   missing fracs  : 0.0 0.25 0.5 0.75 1.0          (5)
#   feature ratios : 1e-5 1e-4 0.001 0.01 0.02 0.05 0.1 0.2 0.25 0.5 0.75 1.0   (12)
#   n-patients     : 20 50 100 200 300 400 500      (7)
#   nomics combos  : 31 (all non-empty subsets incl. singletons)
#   outer/inner CV : 5 / 3
#
# Axes run with full HP grid:  missing | ratio
# Axes run with paper defaults: nomics | npatients
#
# Usage:
#   bash make_experiments.sh                # all datasets
#   bash make_experiments.sh TCGA-BRCA      # single dataset

set -euo pipefail

DATA_ROOT="/work/scitas-share/FellayMultiOmic/code/full-test-pipeline/data/processed"

declare -A DATASET_OMICS
DATASET_OMICS["TCGA-BRCA"]="mrna,dnam,rppa,mirna,cnv"
DATASET_OMICS["TCGA-LGG"]="mrna,dnam,rppa,mirna,cnv"
DATASET_OMICS["TCGA-KIPAN"]="mrna,dnam,rppa,mirna,cnv"

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=("TCGA-BRCA" "TCGA-LGG" "TCGA-KIPAN")
fi

# ── HP grid (P-NET specific) ─────────────────────────────────────────────────
HUS=(64 128)
DOS=(0.1 0.3 0.5)
WRS=(0.001 0.01 0.1)
KS=(50 100 200)        # missing axis only (our feature-selection HP)

# ── Shared benchmark standard ────────────────────────────────────────────────
SEEDS=(0 1 2)
FRACS=(0.0 0.25 0.5 0.75 1.0)
RATIOS=(1e-05 0.0001 0.001 0.01 0.02 0.05 0.1 0.2 0.25 0.5 0.75 1.0)
NPATS=(20 50 100 200 300 400 500)

# Paper defaults (for nomics / npatients axes)
HU_DEF=64; DO_DEF=0.5; WR_DEF=0.001; K_DEF=100

# ── Omic combinations (nomics axis): all 31 non-empty subsets of 5 omics ──────
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

# ── Generate ─────────────────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}"; do
  DATASET_PATH="${DATA_ROOT}/${DATASET}.joblib"
  ALL_OMICS="${DATASET_OMICS[$DATASET]}"
  COMMON="--dataset-path ${DATASET_PATH} --outer-splits 5 --epochs 100"

  # ── MISSING axis: full HP grid ──────────────────────────────────────────
  OUTFILE="experiments_${DATASET}_missing_hpgrid.txt"
  : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do
    for hu in "${HUS[@]}"; do
      for do_ in "${DOS[@]}"; do
        for wr in "${WRS[@]}"; do
          for k in "${KS[@]}"; do
            for frac in "${FRACS[@]}"; do
              echo "python run_missing.py ${COMMON} --omics ${ALL_OMICS} --hidden-units ${hu} --dropout ${do_} --w-reg ${wr} --k-per-omic ${k} --include-non-intersection-frac ${frac} --random-state ${seed} --output-root results/${DATASET}/missing" >> "$OUTFILE"
            done
          done
        done
      done
    done
  done
  echo "${DATASET} missing hpgrid: $(wc -l < "$OUTFILE") jobs → ${OUTFILE}"

  # ── RATIO axis: full HP grid ────────────────────────────────────────────
  OUTFILE="experiments_${DATASET}_ratio_hpgrid.txt"
  : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do
    for hu in "${HUS[@]}"; do
      for do_ in "${DOS[@]}"; do
        for wr in "${WRS[@]}"; do
          for ratio in "${RATIOS[@]}"; do
            echo "python run_ratio.py ${COMMON} --omics ${ALL_OMICS} --hidden-units ${hu} --dropout ${do_} --w-reg ${wr} --ratio-per-omic ${ratio} --include-non-intersection-frac 1.0 --random-state ${seed} --output-root results/${DATASET}/ratio" >> "$OUTFILE"
          done
        done
      done
    done
  done
  echo "${DATASET} ratio hpgrid: $(wc -l < "$OUTFILE") jobs → ${OUTFILE}"

  # ── NOMICS axis: paper defaults only ────────────────────────────────────
  OUTFILE="experiments_${DATASET}_nomics_defaults.txt"
  : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do
    for combo in "${COMBOS[@]}"; do
      echo "python run_nomics.py ${COMMON} --omics ${combo} --hidden-units ${HU_DEF} --dropout ${DO_DEF} --w-reg ${WR_DEF} --k-per-omic ${K_DEF} --include-non-intersection-frac 1.0 --random-state ${seed} --output-root results/${DATASET}/nomics" >> "$OUTFILE"
    done
  done
  echo "${DATASET} nomics defaults: $(wc -l < "$OUTFILE") jobs → ${OUTFILE}"

  # ── NPATIENTS axis: paper defaults only ─────────────────────────────────
  OUTFILE="experiments_${DATASET}_npatients_defaults.txt"
  : > "$OUTFILE"
  for seed in "${SEEDS[@]}"; do
    for n in "${NPATS[@]}"; do
      echo "python run_npatients.py ${COMMON} --omics ${ALL_OMICS} --hidden-units ${HU_DEF} --dropout ${DO_DEF} --w-reg ${WR_DEF} --k-per-omic ${K_DEF} --n-patients ${n} --include-non-intersection-frac 1.0 --random-state ${seed} --output-root results/${DATASET}/npatients" >> "$OUTFILE"
    done
  done
  echo "${DATASET} npatients defaults: $(wc -l < "$OUTFILE") jobs → ${OUTFILE}"

done
