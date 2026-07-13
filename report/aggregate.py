"""Stage-2 aggregation -> the required output structure of
../README_multiomics_benchmark_evaluation.md (section 14).

Reads report/canonical/{fold_metrics,per_class,predictions}/*.pkl.gz and writes,
under report/:

  results/                 machine-readable per-experiment CSVs
    dataset_characteristics/  <DS>_characteristics.csv
    predictions/              per-dataset held-out predictions
    fold_metrics/             all per-fold metrics (long form)
    aggregated_metrics/       main full-data performance
    omic_combinations/        combination leaderboard
    patient_sweep/            npatients axis
    missingness/              missing axis
    feature_sweep/            ratio axis
    statistical_tests/        pairwise comparisons
    computational_cost/       runtimes
  figures/main/            fig01,02,03,05,08,09,12,16  (.png + .pdf)
  figures/supplementary/   all_metrics, all_omic_combinations, calibration,
                           learning_curve_stability, missing_omic_identity,
                           feature_sensitivity, computational_cost
  tables/                  table01 .. table17  (exact spec names)

Aggregation follows README section 3.4 (mean over folds within a repeat, then
mean +/- SD across repeats), balanced-accuracy-primary ranking (4.5), and the
section 11 paired comparisons. Robust to partial input. Run in the `mofa` conda
env (numpy 2.x + pyarrow read every shard).

Usage: python aggregate.py
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CANON = os.path.join(HERE, "canonical")
RES = os.path.join(HERE, "results")
FIG_MAIN = os.path.join(HERE, "figures", "main")
FIG_SUPP = os.path.join(HERE, "figures", "supplementary")
TAB = os.path.join(HERE, "tables")

_RES_SUB = ["dataset_characteristics", "predictions", "fold_metrics",
            "aggregated_metrics", "omic_combinations", "patient_sweep",
            "missingness", "feature_sweep", "statistical_tests",
            "computational_cost"]
_SUPP_SUB = ["all_metrics", "all_omic_combinations", "calibration",
             "learning_curve_stability", "missing_omic_identity",
             "feature_sensitivity", "computational_cost"]
for d in [RES, FIG_MAIN, FIG_SUPP, TAB]:
    os.makedirs(d, exist_ok=True)
for s in _RES_SUB:
    os.makedirs(os.path.join(RES, s), exist_ok=True)
for s in _SUPP_SUB:
    os.makedirs(os.path.join(FIG_SUPP, s), exist_ok=True)

PRIMARY = ["balanced_accuracy", "macro_f1"]
SECONDARY = ["accuracy", "macro_precision", "macro_recall", "weighted_f1", "mcc",
             "cohen_kappa", "macro_roc_auc_ovr", "weighted_roc_auc_ovr",
             "macro_pr_auc_ovr", "log_loss", "brier_score", "ece"]
METRICS = PRIMARY + SECONDARY

MODELS = ["logreg_early", "svm_early", "pca_logreg", "logreg_late", "pnet",
          "mofa", "integrao"]
_OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
COLOR = {m: _OKABE[i % len(_OKABE)] for i, m in enumerate(MODELS)}
PRETTY = {"logreg_early": "LogReg (early)", "svm_early": "SVM (early)",
          "pca_logreg": "PCA+LogReg", "logreg_late": "LogReg (late)",
          "pnet": "P-NET", "mofa": "MOFA", "integrao": "IntegrAO"}
DATASETS = ["TCGA-BRCA", "TCGA-LGG", "TCGA-KIPAN"]
ALL_OMICS = ["mrna", "dnam", "rppa", "mirna", "cnv"]


# --------------------------------------------------------------------------- IO
def _load(kind: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(CANON, kind, "*.pkl.gz"))):
        try:
            d = pd.read_pickle(f)
            if len(d):
                frames.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {os.path.basename(f)}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for c in ("dataset", "model", "omic_combination"):
        if c in df:
            df[c] = df[c].astype(str)
    return df


def _dataset_summaries() -> dict:
    """One dataset summary per dataset, read from any run's metadata.json."""
    out = {}
    for ds in DATASETS:
        for mdl in MODELS:
            fs = glob.glob(os.path.join(ROOT, mdl, "results", ds, "*", "*", "*", "*",
                                        "metadata.json"))
            fs = fs or glob.glob(os.path.join(ROOT, mdl, f"results_{mdl}*", ds, "*",
                                              "*", "metadata.json"))
            if not fs:
                continue
            try:
                meta = json.load(open(fs[0]))
                summ = meta.get("dataset", {}).get("summary")
                if summ:
                    out[ds] = summ
                    break
            except Exception:  # noqa: BLE001
                continue
    return out


# ------------------------------------------------------------------ aggregation
def agg(df: pd.DataFrame, group: list, metrics=METRICS):
    metrics = [m for m in metrics if m in df.columns]
    per_seed = df.groupby(group + ["seed"], dropna=False)[metrics].mean().reset_index()
    g = per_seed.groupby(group, dropna=False)
    out = (g[metrics].mean().add_suffix("_mean")
           .join(g[metrics].std(ddof=1).add_suffix("_sd"))
           .join(g.size().rename("n_units")).reset_index())
    return out, per_seed


def _fmt(m, s):
    if pd.isna(m):
        return ""
    return f"{m:.3f} +/- {s:.3f}" if not pd.isna(s) else f"{m:.3f}"


def pretty_table(a, id_cols, metrics):
    out = a[id_cols].copy()
    for m in metrics:
        if f"{m}_mean" in a:
            out[m] = [_fmt(mu, sd) for mu, sd in zip(a[f"{m}_mean"], a[f"{m}_sd"])]
    if "n_units" in a:
        out["n_evaluation_units"] = a["n_units"].values
    return out


def full_data_mask(df):
    keep = []
    for _, g in df.groupby(["dataset", "model"]):
        m = g[g["n_omics"] == g["n_omics"].max()]
        if m["include_non_intersection_frac"].notna().any():
            m = m[m["include_non_intersection_frac"] == m["include_non_intersection_frac"].max()]
        if m["n_patients"].notna().any():
            m = m[(m["n_patients"] == m["n_patients"].max()) | m["n_patients"].isna()]
        keep.append(m)
    out = pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]
    return out.drop_duplicates(subset=["dataset", "model", "run_name", "seed", "fold"])


# ---------------------------------------------------------------------- figures
def save_fig(fig, where, name):
    fig.tight_layout()
    base = os.path.join(where, name)
    fig.savefig(base + ".png", bbox_inches="tight", dpi=150)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _series_vs(ax, sub, xcol, metric):
    for m in MODELS:
        md = sub[sub["model"] == m]
        if md.empty:
            continue
        a, _ = agg(md, [xcol], [metric])
        a = a.sort_values(xcol)
        x = a[xcol].astype(float).values
        mu = a[f"{metric}_mean"].values
        sd = a[f"{metric}_sd"].fillna(0).values
        ax.plot(x, mu, "-o", ms=3.5, color=COLOR[m], label=PRETTY[m])
        ax.fill_between(x, mu - sd, mu + sd, color=COLOR[m], alpha=0.15)


def grid_metric_vs_axis(df, xcol, xlabel, title, where, name, logx=False):
    """Rows = primary metrics, cols = datasets; mean +/- SD ribbons."""
    sub = df[df[xcol].notna()]
    if sub.empty:
        return
    fig, axes = plt.subplots(len(PRIMARY), len(DATASETS),
                             figsize=(5 * len(DATASETS), 4 * len(PRIMARY)),
                             dpi=150, squeeze=False)
    for i, metric in enumerate(PRIMARY):
        for j, ds in enumerate(DATASETS):
            ax = axes[i][j]
            _series_vs(ax, sub[sub["dataset"] == ds], xcol, metric)
            if logx:
                ax.set_xscale("log")
            if i == len(PRIMARY) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(metric.replace("_", " "))
            if i == 0:
                ax.set_title(ds)
            ax.grid(True, alpha=0.25)
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.suptitle(title, y=1.005)
    save_fig(fig, where, name)


# ----------------------------------------------------------------------- tables
def table01_characteristics(pc, summaries):
    rows = []
    for ds in DATASETS:
        s = summaries.get(ds)
        if not s:
            continue
        total = s.get("n_patients")
        cc = s.get("class_counts", {})
        fpo = s.get("features_per_omic", {})
        mro = s.get("missing_rate_per_omic", {})
        omics = s.get("omics", ALL_OMICS)
        ntot = sum(cc.values()) or total or 1
        maxlen = max(len(cc), len(omics))
        for i in range(maxlen):
            row = {"dataset": ds, "prediction_target": "molecular_subtype"}
            if i < len(cc):
                cls = list(cc.keys())[i]
                row.update({"class": cls, "class_count": cc[cls],
                            "class_percentage": round(100 * cc[cls] / ntot, 2)})
            if i == 0:
                row["total_eligible_patients"] = total
            if i < len(omics):
                om = omics[i]
                obs = int(round(total * (1 - mro.get(om, 0)))) if total else None
                row.update({"omic_name": om, "raw_feature_count": fpo.get(om),
                            "observed_patients": obs,
                            "complete_case_count": obs})
            rows.append(row)
        # per-dataset characteristics file too
        pd.DataFrame([r for r in rows if r["dataset"] == ds]).to_csv(
            os.path.join(RES, "dataset_characteristics",
                         f"{ds.replace('-', '_')}_characteristics.csv"), index=False)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(TAB, "table01_dataset_characteristics.csv"),
                                  index=False)
        print("  table01_dataset_characteristics")


def table02_main(fm):
    fd = full_data_mask(fm)
    if fd.empty:
        return None
    a, _ = agg(fd, ["dataset", "model", "integration", "omic_combination", "n_omics"])
    show = ["balanced_accuracy", "macro_f1", "accuracy", "mcc", "macro_roc_auc_ovr",
            "macro_pr_auc_ovr", "log_loss", "brier_score", "ece"]
    t = pretty_table(a, ["dataset", "model", "integration", "omic_combination", "n_omics"], show)
    t.rename(columns={"omic_combination": "omics_used", "n_omics": "number_of_omics",
                      "integration": "integration_type", "model": "method"}, inplace=True)
    t.sort_values(["dataset", "method"]).to_csv(
        os.path.join(TAB, "table02_main_model_performance.csv"), index=False)
    a.to_csv(os.path.join(RES, "aggregated_metrics", "main_full_data_performance.csv"),
             index=False)
    print("  table02_main_model_performance")
    # Figure 1 — overall performance (both primary metrics, all datasets)
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(7 * len(PRIMARY), 5), dpi=150,
                             squeeze=False)
    present = [m for m in MODELS if m in fd["model"].unique()]
    width = 0.11
    for k, metric in enumerate(PRIMARY):
        ax = axes[0][k]
        for i, m in enumerate(present):
            aa, _ = agg(fd[fd["model"] == m], ["dataset"], [metric])
            aa = aa.set_index("dataset").reindex(DATASETS)
            xs = np.arange(len(DATASETS)) + i * width
            ax.bar(xs, aa[f"{metric}_mean"].values, width, yerr=aa[f"{metric}_sd"].values,
                   color=COLOR[m], label=PRETTY[m], capsize=2)
        ax.set_xticks(np.arange(len(DATASETS)) + width * (len(present) - 1) / 2)
        ax.set_xticklabels(DATASETS)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric.replace("_", " "))
        ax.grid(True, axis="y", alpha=0.25)
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle("Overall full-data performance across datasets", y=1.02)
    save_fig(fig, FIG_MAIN, "fig01_overall_performance")
    return fd


def table03_perclass(pc, fd_selected):
    if pc.empty:
        return
    fd = full_data_mask(pc)
    if fd.empty:
        return
    mets = ["precision", "recall", "specificity", "f1", "roc_auc_ovr", "pr_auc_ovr"]
    a, _ = agg(fd, ["dataset", "model", "class"], mets)
    sup = fd.groupby(["dataset", "model", "class"])[["support", "prevalence"]].mean().reset_index()
    a = a.merge(sup, on=["dataset", "model", "class"], how="left")
    t = pretty_table(a, ["dataset", "model", "class"], mets)
    t.insert(3, "support", a["support"].round(1).values)
    t.insert(4, "prevalence", a["prevalence"].round(3).values)
    t.rename(columns={"model": "method"}, inplace=True)
    t.to_csv(os.path.join(TAB, "table03_per_class_results.csv"), index=False)
    print("  table03_per_class_results")
    # Figure 2 — per-class performance (F1 heatmap, one panel per dataset)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.5 * len(DATASETS), 4.5),
                             dpi=150, squeeze=False)
    for j, ds in enumerate(DATASETS):
        ax = axes[0][j]
        d = a[a["dataset"] == ds]
        piv = d.pivot_table(index="model", columns="class", values="f1_mean").reindex(
            [m for m in MODELS if m in d["model"].unique()])
        if piv.empty:
            ax.axis("off"); continue
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45,
                                                               ha="right", fontsize=7)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([PRETTY.get(m, m) for m in piv.index], fontsize=8)
        ax.set_title(ds)
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                v = piv.values[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=6)
    fig.colorbar(im, ax=axes[0][-1], fraction=0.046, label="per-class F1")
    fig.suptitle("Per-class F1 (full-data)", y=1.02)
    save_fig(fig, FIG_MAIN, "fig02_per_class_performance")


def confusion(pred):
    if pred.empty:
        return
    fd = full_data_mask(pred)
    if fd.empty:
        return
    fd = fd.copy()
    fd["true_label"] = fd["true_label"].astype(str)
    fd["pred_label"] = fd["pred_label"].astype(str)
    # Figure 3 (main) — confusion matrix of the selected best model per dataset
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 3.8),
                             dpi=150, squeeze=False)
    for j, ds in enumerate(DATASETS):
        ax = axes[0][j]
        d = fd[fd["dataset"] == ds]
        if d.empty:
            ax.axis("off"); continue
        best = (d.groupby("model").apply(lambda g: (g.true_label == g.pred_label).mean())
                .sort_values(ascending=False))
        m = best.index[0]
        dm = d[d["model"] == m]
        classes = sorted(set(dm.true_label) | set(dm.pred_label))
        idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)))
        for t, p in zip(dm.true_label, dm.pred_label):
            cm[idx[t], idx[p]] += 1
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{ds}\n{PRETTY.get(m, m)}", fontsize=9)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=6)
        for a_ in range(len(classes)):
            for b_ in range(len(classes)):
                ax.text(b_, a_, f"{cmn[a_, b_]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if cmn[a_, b_] > 0.5 else "black")
    fig.suptitle("Confusion matrices — selected model per dataset", y=1.02)
    save_fig(fig, FIG_MAIN, "fig03_confusion_matrices")
    # supplementary: every model
    for ds in DATASETS:
        d = fd[fd["dataset"] == ds]
        models = [m for m in MODELS if m in d["model"].unique()]
        if not models:
            continue
        classes = sorted(set(d.true_label) | set(d.pred_label))
        idx = {c: i for i, c in enumerate(classes)}
        cols = min(4, len(models)); rows = int(np.ceil(len(models) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows), dpi=130,
                                 squeeze=False)
        for k, m in enumerate(models):
            ax = axes[k // cols][k % cols]
            dm = d[d["model"] == m]
            cm = np.zeros((len(classes), len(classes)))
            for t, p in zip(dm.true_label, dm.pred_label):
                cm[idx[t], idx[p]] += 1
            ax.imshow(cm / cm.sum(1, keepdims=True).clip(min=1), cmap="Blues", vmin=0, vmax=1)
            ax.set_title(PRETTY.get(m, m), fontsize=8)
            ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=90, fontsize=5)
            ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=5)
        for k in range(len(models), rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(f"Confusion matrices — {ds}", y=1.0)
        save_fig(fig, os.path.join(FIG_SUPP, "all_metrics"), f"confusion_all_{ds}")


def experiment_A(fm):
    d = fm[fm["axis"] == "npatients"]
    if d.empty:
        return
    a, _ = agg(d, ["dataset", "model", "n_patients"])
    a.to_csv(os.path.join(RES, "patient_sweep", "patient_sweep.csv"), index=False)
    t = pretty_table(a, ["dataset", "model", "n_patients"],
                     PRIMARY + ["accuracy", "mcc", "macro_roc_auc_ovr"])
    t.rename(columns={"model": "method"}, inplace=True)
    t.sort_values(["dataset", "method", "n_patients"]).to_csv(
        os.path.join(TAB, "table04_patient_count_sweep.csv"), index=False)
    print("  table04_patient_count_sweep")
    grid_metric_vs_axis(d, "n_patients", "number of patients",
                        "Primary metrics vs number of patients", FIG_MAIN,
                        "fig05_metrics_vs_patients")
    # Figure 6 (supp) — stability: SD across seeds vs n_patients
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for m in MODELS:
        md = a[a["model"] == m]
        if md.empty:
            continue
        g = md.groupby("n_patients")["balanced_accuracy_sd"].mean().reset_index()
        ax.plot(g["n_patients"], g["balanced_accuracy_sd"], "-o", ms=3.5,
                color=COLOR[m], label=PRETTY[m])
    ax.set_xlabel("number of patients"); ax.set_ylabel("SD of balanced accuracy (across seeds)")
    ax.set_title("Stability vs number of patients"); ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, os.path.join(FIG_SUPP, "learning_curve_stability"), "fig06_stability_vs_patients")


def experiment_B(fm):
    d = fm[fm["axis"] == "nomics"]
    if d.empty:
        return
    a, _ = agg(d, ["dataset", "model", "n_omics", "omic_combination"])
    a.to_csv(os.path.join(RES, "omic_combinations", "combination_leaderboard.csv"), index=False)
    # Table 6 — all combinations
    lb = pretty_table(a, ["dataset", "model", "n_omics", "omic_combination"],
                      PRIMARY + ["accuracy", "mcc"])
    lb.rename(columns={"model": "method"}, inplace=True)
    lb.sort_values(["dataset", "method", "n_omics"]).to_csv(
        os.path.join(TAB, "table06_all_omic_combinations.csv"), index=False)
    print("  table06_all_omic_combinations")
    # Table 5 — best combination by number of omics
    best = (a.sort_values("balanced_accuracy_mean", ascending=False)
              .groupby(["dataset", "model", "n_omics"], as_index=False).first())
    b = pretty_table(best, ["dataset", "model", "n_omics", "omic_combination"],
                     PRIMARY + ["accuracy"])
    b.rename(columns={"model": "method", "omic_combination": "best_combination"}, inplace=True)
    b.sort_values(["dataset", "method", "n_omics"]).to_csv(
        os.path.join(TAB, "table05_best_combination_by_n_omics.csv"), index=False)
    print("  table05_best_combination_by_n_omics")
    # Table 9 — complete omic absence: full-5 vs each 4-omic (omic X removed)
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            g = a[(a.dataset == ds) & (a.model == m)]
            full = g[g.n_omics == 5]
            if full.empty:
                continue
            base = full["balanced_accuracy_mean"].iloc[0]
            basef = full["macro_f1_mean"].iloc[0]
            for om in ALL_OMICS:
                present = [o for o in ALL_OMICS if o != om]
                match = g[(g.n_omics == 4) & g.omic_combination.apply(
                    lambda c: set(c.split("+")) == set(present))]
                if match.empty:
                    continue
                rows.append({"dataset": ds, "method": m, "omic_removed": om,
                             "balanced_accuracy": round(match["balanced_accuracy_mean"].iloc[0], 4),
                             "delta_balanced_accuracy": round(match["balanced_accuracy_mean"].iloc[0] - base, 4),
                             "delta_macro_f1": round(match["macro_f1_mean"].iloc[0] - basef, 4)})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(TAB, "table09_complete_omic_absence.csv"),
                                  index=False)
        print("  table09_complete_omic_absence")
        # Figure 14 (supp) — completely missing individual omics
        dfr = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.2),
                                 dpi=150, squeeze=False)
        for j, ds in enumerate(DATASETS):
            ax = axes[0][j]
            sd = dfr[dfr.dataset == ds]
            piv = sd.pivot_table(index="method", columns="omic_removed",
                                 values="delta_balanced_accuracy").reindex(
                [m for m in MODELS if m in sd.method.unique()])
            if piv.empty:
                ax.axis("off"); continue
            im = ax.imshow(piv.values, cmap="RdBu", vmin=-0.2, vmax=0.2, aspect="auto")
            ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=7)
            ax.set_yticks(range(piv.shape[0]))
            ax.set_yticklabels([PRETTY.get(m, m) for m in piv.index], fontsize=8)
            ax.set_title(ds)
        fig.colorbar(im, ax=axes[0][-1], fraction=0.046, label="Δ balanced accuracy")
        fig.suptitle("Effect of completely removing each omic (vs all-5)", y=1.02)
        save_fig(fig, os.path.join(FIG_SUPP, "missing_omic_identity"),
                 "fig14_complete_omic_absence")
    # Table 7 — nested combination selection: NOT a run experiment in these axes
    _stub("table07_nested_combination_selection.csv",
          ["dataset", "method", "selected_combination", "outer_test_balanced_accuracy",
           "outer_test_macro_f1", "note"],
          "Nested (inner-fold) combination selection was not run as a separate "
          "protocol; the nomics axis evaluates every combination post-hoc "
          "(see table06). Populate by running a nested selection experiment.")
    # Figures 8 / 9 — best performance & best combination vs number of omics
    for metric in PRIMARY:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.2),
                                 dpi=150, squeeze=False)
        for j, ds in enumerate(DATASETS):
            ax = axes[0][j]
            sub = best[best.dataset == ds]
            for m in MODELS:
                md = sub[sub.model == m].sort_values("n_omics")
                if md.empty:
                    continue
                ax.plot(md.n_omics, md[f"{metric}_mean"], "-o", ms=3.5, color=COLOR[m],
                        label=PRETTY[m])
                ax.fill_between(md.n_omics, md[f"{metric}_mean"] - md[f"{metric}_sd"].fillna(0),
                                md[f"{metric}_mean"] + md[f"{metric}_sd"].fillna(0),
                                color=COLOR[m], alpha=0.15)
            ax.set_xlabel("number of omics"); ax.set_title(ds)
            if j == 0:
                ax.set_ylabel(f"best {metric.replace('_', ' ')}")
            ax.grid(True, alpha=0.25)
        axes[0][-1].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Best {metric.replace('_', ' ')} vs number of omics", y=1.02)
        save_fig(fig, FIG_MAIN, f"fig08_best_performance_vs_n_omics_{metric}")
    # Figure 9 — best-combination labels at each k (balanced accuracy), one panel/dataset
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(9, 3.2 * len(DATASETS)),
                             dpi=150, squeeze=False)
    for j, ds in enumerate(DATASETS):
        ax = axes[j][0]
        sub = best[(best.dataset == ds)]
        for m in MODELS:
            md = sub[sub.model == m].sort_values("n_omics")
            if md.empty:
                continue
            ax.plot(md.n_omics, md.balanced_accuracy_mean, "-o", ms=3.5, color=COLOR[m],
                    label=PRETTY[m])
        # annotate the overall best combo per k
        top = (sub.sort_values("balanced_accuracy_mean", ascending=False)
                  .groupby("n_omics", as_index=False).first())
        for _, r in top.iterrows():
            ax.annotate(r.omic_combination, (r.n_omics, r.balanced_accuracy_mean),
                        fontsize=6, xytext=(0, 6), textcoords="offset points", ha="center")
        ax.set_title(f"{ds} — best combination labelled at each k", fontsize=9)
        ax.set_xlabel("number of omics"); ax.set_ylabel("balanced accuracy")
        ax.grid(True, alpha=0.25)
    axes[0][0].legend(fontsize=6, ncol=3)
    save_fig(fig, FIG_MAIN, "fig09_best_combination_by_n_omics")
    # Figure 10 (supp) — all combinations scatter
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.2),
                             dpi=150, squeeze=False)
    for j, ds in enumerate(DATASETS):
        ax = axes[0][j]
        sub = a[a.dataset == ds]
        for m in MODELS:
            md = sub[sub.model == m]
            if md.empty:
                continue
            ax.scatter(md.n_omics + np.random.uniform(-0.12, 0.12, len(md)),
                       md.balanced_accuracy_mean, s=14, color=COLOR[m], alpha=0.6,
                       label=PRETTY[m])
        ax.set_xlabel("number of omics"); ax.set_title(ds)
        if j == 0:
            ax.set_ylabel("balanced accuracy")
        ax.grid(True, alpha=0.25)
    axes[0][-1].legend(fontsize=6, ncol=2)
    fig.suptitle("All omic combinations", y=1.02)
    save_fig(fig, os.path.join(FIG_SUPP, "all_omic_combinations"), "fig10_all_combinations")


def experiment_C(fm):
    d = fm[fm["axis"] == "missing"]
    if d.empty:
        return
    a, _ = agg(d, ["dataset", "model", "include_non_intersection_frac"])
    a.to_csv(os.path.join(RES, "missingness", "random_missingness.csv"), index=False)
    t = pretty_table(a, ["dataset", "model", "include_non_intersection_frac"],
                     PRIMARY + ["accuracy", "mcc", "macro_roc_auc_ovr"])
    t.rename(columns={"model": "method",
                      "include_non_intersection_frac": "included_incomplete_fraction"},
             inplace=True)
    t.to_csv(os.path.join(TAB, "table08_random_missingness.csv"), index=False)
    print("  table08_random_missingness")
    # Table 10 — robustness ranking
    rows = []
    for (ds, m), g in a.groupby(["dataset", "model"]):
        g = g.sort_values("include_non_intersection_frac")
        x = g["include_non_intersection_frac"].astype(float).values
        y = g["balanced_accuracy_mean"].values
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else np.nan
        rows.append({"dataset": ds, "method": m,
                     "balanced_accuracy_full_overlap": round(y[np.argmin(x)], 4),
                     "balanced_accuracy_most_incomplete": round(y[np.argmax(x)], 4),
                     "drop": round(y[np.argmin(x)] - y[np.argmax(x)], 4),
                     "slope_per_unit_frac": round(slope, 4)})
    pd.DataFrame(rows).sort_values(["dataset", "drop"]).to_csv(
        os.path.join(TAB, "table10_missingness_robustness.csv"), index=False)
    print("  table10_missingness_robustness")
    grid_metric_vs_axis(d, "include_non_intersection_frac",
                        "included incomplete fraction",
                        "Primary metrics vs random omic missingness", FIG_MAIN,
                        "fig12_metrics_vs_missingness")


def experiment_D(fm):
    d = fm[fm["axis"] == "ratio"]
    if d.empty:
        return
    xcol = "ratio_per_omic" if d["ratio_per_omic"].notna().any() else "k_per_omic"
    a, _ = agg(d, ["dataset", "model", xcol])
    a.to_csv(os.path.join(RES, "feature_sweep", "feature_percentage_sweep.csv"), index=False)
    t = pretty_table(a, ["dataset", "model", xcol], PRIMARY + ["accuracy", "mcc"])
    t.rename(columns={"model": "method", xcol: "feature_fraction_per_omic"}, inplace=True)
    t.to_csv(os.path.join(TAB, "table11_feature_percentage_sweep.csv"), index=False)
    print("  table11_feature_percentage_sweep")
    # Table 13 — feature efficiency
    rows = []
    for (ds, m), g in a.groupby(["dataset", "model"]):
        g = g.sort_values(xcol)
        best = g["balanced_accuracy_mean"].max()
        bsd = g.loc[g["balanced_accuracy_mean"].idxmax(), "balanced_accuracy_sd"]
        thr = best - (0 if pd.isna(bsd) else bsd)
        pick = g[g["balanced_accuracy_mean"] >= thr].iloc[0]
        rows.append({"dataset": ds, "method": m,
                     "smallest_feature_fraction_within_1sd": pick[xcol],
                     "balanced_accuracy_at_that_point": round(pick["balanced_accuracy_mean"], 4),
                     "macro_f1_at_that_point": round(pick["macro_f1_mean"], 4),
                     "best_balanced_accuracy": round(best, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "table13_feature_efficiency.csv"), index=False)
    print("  table13_feature_efficiency")
    grid_metric_vs_axis(d, xcol, "feature fraction per omic (log)",
                        "Primary metrics vs percentage of features", FIG_MAIN,
                        "fig16_metrics_vs_feature_percentage", logx=True)
    # Table 12 — per-omic one-at-a-time sensitivity: not present in these axes
    _stub("table12_per_omic_feature_sensitivity.csv",
          ["dataset", "method", "omic", "feature_fraction", "balanced_accuracy",
           "macro_f1", "note"],
          "The feature-percentage axis is a JOINT sweep over all omics "
          "(all-5, single combination). A one-omic-at-a-time feature sweep was "
          "not run, so per-omic sensitivity cannot be derived from these results.")


def cross_dataset(fm):
    fd = full_data_mask(fm)
    if fd.empty:
        return
    a, _ = agg(fd, ["dataset", "model"])
    a["rank"] = a.groupby("dataset")["balanced_accuracy_mean"].rank(ascending=False)
    summ = a.groupby("model").agg(
        mean_balanced_accuracy=("balanced_accuracy_mean", "mean"),
        mean_macro_f1=("macro_f1_mean", "mean"),
        mean_rank=("rank", "mean"),
        n_datasets=("dataset", "nunique")).reset_index().sort_values("mean_rank")
    summ.rename(columns={"model": "method"}, inplace=True)
    summ.round(4).to_csv(os.path.join(TAB, "table14_cross_dataset_summary.csv"), index=False)
    print("  table14_cross_dataset_summary")
    win = (a.sort_values("balanced_accuracy_mean", ascending=False)
             .groupby("dataset", as_index=False).first()[
                 ["dataset", "model", "balanced_accuracy_mean", "macro_f1_mean"]])
    win.rename(columns={"model": "winning_method"}, inplace=True)
    win.round(4).to_csv(os.path.join(TAB, "table15_dataset_winners.csv"), index=False)
    print("  table15_dataset_winners")
    # Figure 20 (supp) — method rank across datasets
    piv = a.pivot_table(index="model", columns="dataset", values="rank").reindex(
        [m for m in MODELS if m in a.model.unique()])
    fig, ax = plt.subplots(figsize=(1.6 * len(DATASETS) + 3, 0.6 * piv.shape[0] + 2), dpi=150)
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([PRETTY.get(m, m) for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center")
    ax.set_title("Method rank by balanced accuracy (1 = best)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="rank")
    save_fig(fig, os.path.join(FIG_SUPP, "all_metrics"), "fig21_method_rank")


def cost(fm):
    fd = full_data_mask(fm)
    if fd.empty or "runtime_seconds" not in fd:
        return
    per_seed = fd.groupby(["dataset", "model", "seed"]).agg(
        bacc=("balanced_accuracy", "mean"), rt=("runtime_seconds", "mean")).reset_index()
    g = per_seed.groupby(["dataset", "model"]).agg(
        balanced_accuracy=("bacc", "mean"),
        runtime_seconds_per_fold=("rt", "mean")).reset_index()
    g.round(4).to_csv(os.path.join(RES, "computational_cost", "computational_cost.csv"),
                      index=False)
    g.rename(columns={"model": "method"}).round(4).to_csv(
        os.path.join(TAB, "table17_computational_cost.csv"), index=False)
    print("  table17_computational_cost")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for m in MODELS:
        md = g[g.model == m]
        if md.empty:
            continue
        ax.scatter(md.runtime_seconds_per_fold, md.balanced_accuracy, color=COLOR[m],
                   label=PRETTY[m], s=45)
    ax.set_xscale("log"); ax.set_xlabel("runtime per fold (s, log)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Performance vs computational cost (full-data)")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=7, ncol=2)
    save_fig(fig, os.path.join(FIG_SUPP, "computational_cost"), "fig22_performance_vs_cost")


def calibration(pred):
    """Figure 4 (supp) — reliability curves (full-data), per dataset."""
    if pred.empty:
        return
    fd = full_data_mask(pred)
    proba_cols = [c for c in fd.columns if c.startswith("proba_")]
    if fd.empty or not proba_cols:
        return
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.5),
                             dpi=150, squeeze=False)
    for j, ds in enumerate(DATASETS):
        ax = axes[0][j]
        d = fd[fd.dataset == ds]
        for m in MODELS:
            dm = d[d.model == m]
            if dm.empty or dm[proba_cols].isna().all().all():
                continue
            conf = dm[proba_cols].max(axis=1).values
            pred_lab = [c[len("proba_"):] for c in proba_cols]
            pick = dm[proba_cols].values.argmax(axis=1)
            predicted = np.array([pred_lab[i] for i in pick])
            correct = (predicted == dm.true_label.astype(str).values).astype(float)
            bins = np.linspace(0, 1, 11)
            xs, ys = [], []
            for b in range(10):
                mask = (conf > bins[b]) & (conf <= bins[b + 1])
                if mask.sum() > 5:
                    xs.append(conf[mask].mean()); ys.append(correct[mask].mean())
            if xs:
                ax.plot(xs, ys, "-o", ms=3, color=COLOR[m], label=PRETTY[m])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("mean predicted confidence"); ax.set_title(ds)
        if j == 0:
            ax.set_ylabel("empirical accuracy")
        ax.grid(True, alpha=0.25)
    axes[0][-1].legend(fontsize=6, ncol=2)
    fig.suptitle("Calibration (reliability) — full-data", y=1.02)
    save_fig(fig, os.path.join(FIG_SUPP, "calibration"), "fig04_calibration")


def _stub(fname, cols, note):
    df = pd.DataFrame([{c: "" for c in cols}])
    df.loc[0, "note"] = note
    df.to_csv(os.path.join(TAB, fname), index=False)
    print(f"  {fname[:-4]} (stub: {note[:48]}...)")


def _export_results_tree(fm, pc, pred):
    """Populate results/fold_metrics and results/predictions (long form)."""
    if not fm.empty:
        fm.to_csv(os.path.join(RES, "fold_metrics", "all_fold_metrics.csv"), index=False)
    if not pred.empty:
        for ds in DATASETS:
            d = pred[pred.dataset == ds]
            if not d.empty:
                d.to_csv(os.path.join(RES, "predictions",
                                      f"{ds.replace('-', '_')}_predictions.csv"), index=False)


def main():
    print("loading canonical shards ...")
    fm = _load("fold_metrics"); pc = _load("per_class"); pred = _load("predictions")
    if fm.empty:
        print("no fold_metrics shards found; nothing to aggregate yet.")
        return
    cells = fm[["dataset", "model", "axis"]].drop_duplicates()
    print(f"fold_metrics: {len(fm)} rows across {len(cells)}/84 cells")
    summaries = _dataset_summaries()
    print("writing results tree, tables, figures ...")
    _export_results_tree(fm, pc, pred)
    table01_characteristics(pc, summaries)
    fd = table02_main(fm)
    table03_perclass(pc, fd)
    confusion(pred)
    experiment_A(fm)
    experiment_B(fm)
    experiment_C(fm)
    experiment_D(fm)
    cross_dataset(fm)
    cost(fm)
    calibration(pred)
    # statistical tests (Table 16)
    stats(fm)
    tabs = sorted(os.path.basename(x) for x in glob.glob(os.path.join(TAB, "*.csv")))
    figs = sorted(os.path.relpath(x, HERE) for x in
                  glob.glob(os.path.join(HERE, "figures", "**", "*.png"), recursive=True))
    json.dump({"tables": tabs, "figures": figs,
               "cells_present": cells.to_dict("records")},
              open(os.path.join(HERE, "STAGE2_MANIFEST.json"), "w"), indent=2, default=str)
    print(f"done: {len(tabs)}/17 tables, {len(figs)} figures.")


def stats(fm):
    from itertools import combinations
    from scipy.stats import wilcoxon
    fd = full_data_mask(fm)
    if fd.empty:
        return
    per_seed = fd.groupby(["dataset", "model", "seed"])[PRIMARY].mean().reset_index()
    rows = []
    for metric in PRIMARY:
        piv = per_seed.pivot_table(index=["dataset", "seed"], columns="model", values=metric)
        models = [m for m in MODELS if m in piv.columns]
        for a_, b_ in combinations(models, 2):
            pair = piv[[a_, b_]].dropna()
            if len(pair) < 2:
                continue
            diff = (pair[a_] - pair[b_]).values
            rng = np.random.default_rng(0)
            boot = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            try:
                p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            except Exception:
                p = np.nan
            rows.append({"metric": metric, "method_a": a_, "method_b": b_,
                         "mean_diff": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                         "n_units": len(diff), "wilcoxon_p": p})
    if not rows:
        return
    t = pd.DataFrame(rows)
    t["p_fdr"] = np.nan
    for metric in PRIMARY:
        mask = t.metric == metric
        p = t.loc[mask, "wilcoxon_p"].values
        order = np.argsort(np.where(np.isnan(p), 1.0, p))
        n = len(p); adj = np.full(n, np.nan); prev = 1.0
        for rank, i in enumerate(reversed(order), start=1):
            k = n - rank + 1
            if np.isnan(p[i]):
                continue
            prev = min(prev, p[i] * n / k); adj[i] = prev
        t.loc[mask, "p_fdr"] = adj
    t.round(4).to_csv(os.path.join(TAB, "table16_pairwise_comparisons.csv"), index=False)
    t.round(4).to_csv(os.path.join(RES, "statistical_tests", "pairwise_comparisons.csv"),
                      index=False)
    print("  table16_pairwise_comparisons")


if __name__ == "__main__":
    main()
