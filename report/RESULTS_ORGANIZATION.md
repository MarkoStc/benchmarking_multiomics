# How the benchmark results are organized

This document explains, end to end, where the numbers live and how you get from a
finished SLURM run to the final tables and figures required by
[`../README_multiomics_benchmark_evaluation.md`](../README_multiomics_benchmark_evaluation.md).

There are three layers:

```
  raw runs            ->   Stage 1: canonical shards   ->   Stage 2: tables + figures
  (per SLURM job)          (one tidy row per fold)          (aggregated, ranked, plotted)
  <model>/results/…        report/canonical/…               report/tables/  report/figures/
```

Nothing is ever retrained. Every layer is derived from the fitted models that the
benchmark already saved.

---

## Layer 0 — raw runs (inputs, not committed to git)

Each finished experiment is one directory under a model:

```
<model>/results/<dataset>/<axis>/<dataset>/<ModelClass>/<run_name>/
    metadata.json        # cohort recipe: omics, seed, frac, n_patients, hparams, splits
    fold_results.csv     # per-fold accuracy/bacc + total_time_sec, as logged during the run
    result.json          # run-level summary
    fold_models/
        fold_models.joblib   # the fitted model for each of the 5 outer folds + train/test idx
```

- **`<model>`** ∈ `logreg_early`, `svm_early`, `pca_logreg`, `logreg_late`, `pnet`,
  `mofa`, `integrao`.
- **`<dataset>`** ∈ `TCGA-BRCA`, `TCGA-LGG`, `TCGA-KIPAN`.
- **`<axis>`** ∈ `npatients`, `ratio`, `nomics`, `missing` — the four benchmark
  sweeps (Experiments A–D in the README).

A run directory exists **only if that config finished**. These trees are large
(fitted models + TCGA matrices) and are **git-ignored** — they are not part of the
repository.

---

## Layer 1 — canonical shards (Stage 1 output)

**Produced by:** `report/harvest.py` (one job per `(model, dataset, axis)`, driven by
`report/manifest.txt` + `report/slurm_harvest.sh`).

For every run, `harvest.py` reconstructs the held-out test predictions
(`report/reconstruct.py`) — a **forward pass** of the saved model on its stored
`test_idx`, never a retrain — and computes the full metric battery
(`report/metrics.py`). Correctness is asserted by re-deriving the fold split and
matching the stored accuracy to ~1e-16 (`report/validate_reconstruction.py`).

Output lives in **`report/canonical/`** as gzipped pandas pickles
(`.pkl.gz`; pickle rather than parquet because the compute envs have no parquet
engine). One file per `(model, dataset, axis)`, in three kinds:

```
report/canonical/
    fold_metrics/<model>__<dataset>__<axis>.pkl.gz   # 1 row per fold
    per_class/  <model>__<dataset>__<axis>.pkl.gz    # 1 row per fold per class
    predictions/<model>__<dataset>__<axis>.pkl.gz    # 1 row per held-out patient
```

These shards are the **single source of truth** for all downstream analysis. They
are git-ignored (large, regenerable).

### Columns

Every shard carries the same **identifying / condition columns**, so any shard can
be concatenated with any other:

| column | meaning |
|---|---|
| `dataset`, `model`, `axis` | which cell of the benchmark |
| `run_name` | the exact run directory name |
| `integration` | integration strategy (early / late / intermediate) |
| `omic_combination`, `n_omics` | which omics, and how many |
| `n_patients` | cohort size (set on the `npatients` axis, else null) |
| `include_non_intersection_frac` | missingness level (the `missing` axis value) |
| `ratio_per_omic`, `k_per_omic` | feature-budget knobs (the `ratio` axis) |
| `seed` | repeat seed (0 / 1 / 2) |
| `hparams` | JSON string of the selected hyper-parameters |
| `fold` | outer fold index (0–4) |

**`fold_metrics`** adds `n_test`, `runtime_seconds`, and the per-fold metrics:
`balanced_accuracy` and `macro_f1` (the README's primary + co-primary),
`accuracy`, `macro_precision`, `macro_recall`, `weighted_f1`, `mcc`,
`cohen_kappa`, and the probability-based `macro_roc_auc_ovr`,
`weighted_roc_auc_ovr`, `macro_pr_auc_ovr`, `log_loss`, `brier_score`, `ece`.
(Probability metrics are `NaN` only if a model truly has no probabilities; SVM gets
training-only Platt-calibrated probabilities per README §4.2, so it is fully
populated.)

**`per_class`** adds, per class: `support`, `prevalence`, `precision`, `recall`,
`specificity`, `f1`, `roc_auc_ovr`, `pr_auc_ovr`, `predicted_frequency`.

**`predictions`** adds, per held-out patient: `patient_id`, `true_label`,
`pred_label`, and one `proba_<class>` column per class — enough to recompute any
metric or confusion matrix from scratch.

---

## Layer 2 — tables and figures (Stage 2 output)

**Produced by:** `report/aggregate.py` (reads all of `report/canonical/`, writes the
deliverables). This is the only layer committed to git.

Aggregation follows README §3.4: average the 5 folds within a run, then average
across the 3 seeds, reporting **mean ± SD** across seeds. Ranking and
hyper-parameter selection use **balanced accuracy** (primary), with **macro-F1** as
co-primary. Paired model comparisons use bootstrap confidence intervals + Wilcoxon
signed-rank with Benjamini–Hochberg FDR correction.

```
report/tables/     # the README's numbered tables, as CSVs (small, committed)
report/figures/    # the README's figures, as PNGs (committed)
```

Each table/figure filename maps to the corresponding numbered item in
§14 of the evaluation README.

---

## TL;DR — regenerating everything from scratch

```bash
# Stage 1 (SLURM): reconstruct predictions + metrics for all 84 (model,dataset,axis) cells
sbatch --array=1-84 report/slurm_harvest.sh          # -> report/canonical/*.pkl.gz

# Stage 2 (single node): aggregate into tables + figures.
# Run in the `mofa` conda env: it has numpy 2.x + pyarrow, so it can read every
# shard (mofa shards were written under numpy 2.x; the rest use Arrow-backed
# string columns that need pyarrow). Safe to re-run any time; it uses whatever
# shards are present and overwrites the tables/figures in place.
conda run -n mofa python report/aggregate.py         # -> report/tables/  report/figures/
```

Aggregation is robust to partial input — `report/STAGE2_MANIFEST.json` records which
of the 84 cells were present in the last run.
