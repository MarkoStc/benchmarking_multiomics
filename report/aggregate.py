"""Stage-2 aggregation: turn the canonical shards into the tables and figures
required by ../README_multiomics_benchmark_evaluation.md.

Reads every shard in report/canonical/{fold_metrics,per_class,predictions}/ and
writes CSV tables to report/tables/ and PNG figures to report/figures/.

Aggregation follows README §3.4: average the outer folds within each repeat
(seed), then report mean +/- SD across the repeat-level means. Ranking and model
selection use balanced accuracy (primary) with macro F1 co-primary (§4.5).

Robust to partial input: whatever shards are on disk are used; missing cells are
simply absent from the outputs. Re-run any time (e.g. once the harvest finishes).

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
CANON = os.path.join(HERE, "canonical")
TAB = os.path.join(HERE, "tables")
FIG = os.path.join(HERE, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# metrics carried through the full battery (fold_metrics shards)
PRIMARY = ["balanced_accuracy", "macro_f1"]
SECONDARY = ["accuracy", "macro_precision", "macro_recall", "weighted_f1", "mcc",
             "cohen_kappa", "macro_roc_auc_ovr", "weighted_roc_auc_ovr",
             "macro_pr_auc_ovr", "log_loss", "brier_score", "ece"]
METRICS = PRIMARY + SECONDARY

# Okabe-Ito colorblind-safe palette, one stable colour per method
MODELS = ["logreg_early", "svm_early", "pca_logreg", "logreg_late", "pnet",
          "mofa", "integrao"]
_OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
COLOR = {m: _OKABE[i % len(_OKABE)] for i, m in enumerate(MODELS)}
PRETTY = {"logreg_early": "LogReg (early)", "svm_early": "SVM (early)",
          "pca_logreg": "PCA+LogReg", "logreg_late": "LogReg (late)",
          "pnet": "P-NET", "mofa": "MOFA", "integrao": "IntegrAO"}
DATASETS = ["TCGA-BRCA", "TCGA-LGG", "TCGA-KIPAN"]


# --------------------------------------------------------------------------- IO
def _load(kind: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(CANON, kind, "*.pkl.gz")))
    frames = []
    for f in files:
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


# ------------------------------------------------------------------ aggregation
def agg(df: pd.DataFrame, group: list, metrics=METRICS):
    """§3.4: mean over folds within each (group, seed), then mean/SD across seeds."""
    metrics = [m for m in metrics if m in df.columns]
    per_seed = df.groupby(group + ["seed"], dropna=False)[metrics].mean().reset_index()
    g = per_seed.groupby(group, dropna=False)
    mean = g[metrics].mean()
    sd = g[metrics].std(ddof=1)
    n = g.size().rename("n_units")
    out = mean.add_suffix("_mean").join(sd.add_suffix("_sd")).join(n).reset_index()
    return out, per_seed


def _fmt(m, s):
    if pd.isna(m):
        return ""
    return f"{m:.3f} +/- {s:.3f}" if not pd.isna(s) else f"{m:.3f}"


def pretty_table(a: pd.DataFrame, id_cols: list, metrics: list) -> pd.DataFrame:
    out = a[id_cols].copy()
    for m in metrics:
        if f"{m}_mean" in a:
            out[m] = [_fmt(mu, sd) for mu, sd in zip(a[f"{m}_mean"], a[f"{m}_sd"])]
    if "n_units" in a:
        out["n_units"] = a["n_units"].values
    return out


def full_data_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Rows at each (dataset, model)'s full-data reference point: all omics, no
    induced missingness (frac==max), max patients, largest feature budget. This
    endpoint appears at the extreme of several axes; dedupe by run_name."""
    keep = []
    for (ds, mdl), g in df.groupby(["dataset", "model"]):
        m = g.copy()
        m = m[m["n_omics"] == m["n_omics"].max()]
        if m["include_non_intersection_frac"].notna().any():
            m = m[m["include_non_intersection_frac"] == m["include_non_intersection_frac"].max()]
        if m["n_patients"].notna().any():
            m = m[(m["n_patients"] == m["n_patients"].max()) | m["n_patients"].isna()]
        keep.append(m)
    out = pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]
    return out.drop_duplicates(subset=["dataset", "model", "run_name", "seed", "fold"])


# ---------------------------------------------------------------------- figures
def _newfig(w=9, h=5.5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    return fig, ax


def _save(fig, name):
    fig.tight_layout()
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {name}")


def line_vs_axis(df, xcol, metric, title, fname, xlabel, logx=False):
    """One panel: metric (mean +/- SD across seeds) vs xcol, one line per model."""
    sub = df[df[xcol].notna()]
    if sub.empty:
        return
    fig, ax = _newfig()
    for m in MODELS:
        md = sub[sub["model"] == m]
        if md.empty:
            continue
        a, _ = agg(md, [xcol], [metric])
        a = a.sort_values(xcol)
        x = a[xcol].astype(float).values
        mu = a[f"{metric}_mean"].values
        sd = a[f"{metric}_sd"].fillna(0).values
        ax.plot(x, mu, "-o", ms=4, color=COLOR[m], label=PRETTY[m])
        ax.fill_between(x, mu - sd, mu + sd, color=COLOR[m], alpha=0.15)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, fname)


# ----------------------------------------------------------------------- tables
def table2_main(fm):
    """Table 2 — main full-data model performance (one row per dataset, method)."""
    fd = full_data_mask(fm)
    if fd.empty:
        return
    a, _ = agg(fd, ["dataset", "model", "integration", "omic_combination", "n_omics"])
    show = ["balanced_accuracy", "macro_f1", "accuracy", "mcc", "macro_roc_auc_ovr",
            "macro_pr_auc_ovr", "log_loss", "brier_score", "ece"]
    t = pretty_table(a, ["dataset", "model", "integration", "omic_combination", "n_omics"], show)
    t = t.sort_values(["dataset", "model"])
    t.to_csv(os.path.join(TAB, "table02_main_model_performance.csv"), index=False)
    print("  table02_main_model_performance")
    # Figure 1 — overall performance across datasets (bars, both primary metrics)
    for metric in PRIMARY:
        fig, ax = _newfig(11, 5.5)
        width = 0.11
        present = [m for m in MODELS if m in fd["model"].unique()]
        for i, m in enumerate(present):
            aa, _ = agg(fd[fd["model"] == m], ["dataset"], [metric])
            aa = aa.set_index("dataset").reindex(DATASETS)
            xs = np.arange(len(DATASETS)) + i * width
            ax.bar(xs, aa[f"{metric}_mean"].values, width, yerr=aa[f"{metric}_sd"].values,
                   color=COLOR[m], label=PRETTY[m], capsize=2)
        ax.set_xticks(np.arange(len(DATASETS)) + width * (len(present) - 1) / 2)
        ax.set_xticklabels(DATASETS)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"Full-data {metric.replace('_', ' ')} across datasets")
        ax.legend(fontsize=8, ncol=4)
        ax.grid(True, axis="y", alpha=0.25)
        _save(fig, f"fig01_overall_{metric}.png")


def table3_perclass(pc):
    """Table 3 — per-class results at the full-data point."""
    if pc.empty:
        return
    fd = full_data_mask(pc)
    if fd.empty:
        return
    metrics = ["precision", "recall", "specificity", "f1", "roc_auc_ovr", "pr_auc_ovr"]
    a, _ = agg(fd, ["dataset", "model", "class"], metrics)
    sup = fd.groupby(["dataset", "model", "class"])[["support", "prevalence"]].mean().reset_index()
    a = a.merge(sup, on=["dataset", "model", "class"], how="left")
    t = pretty_table(a, ["dataset", "model", "class"], metrics)
    t = a[["dataset", "model", "class"]].assign(
        support=a["support"].round(1).values, prevalence=a["prevalence"].round(3).values).merge(
        t, on=["dataset", "model", "class"])
    t.to_csv(os.path.join(TAB, "table03_per_class_results.csv"), index=False)
    print("  table03_per_class_results")
    # Figure 2 — per-class F1 heatmap (one panel per dataset, methods x classes)
    for ds in DATASETS:
        d = a[a["dataset"] == ds]
        if d.empty:
            continue
        piv = d.pivot_table(index="model", columns="class", values="f1_mean")
        piv = piv.reindex([m for m in MODELS if m in piv.index])
        fig, ax = _newfig(max(6, 1 + 0.7 * piv.shape[1]), max(4, 0.6 * piv.shape[0] + 1))
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([PRETTY.get(m, m) for m in piv.index], fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, label="macro-per-class F1")
        ax.set_title(f"Per-class F1 — {ds}")
        _save(fig, f"fig02_perclass_f1_{ds}.png")


def confusion_figs(pred):
    """Figure 3 — confusion matrices (full-data, summed over folds/seeds)."""
    if pred.empty:
        return
    fd = full_data_mask(pred)
    if fd.empty:
        return
    for ds in DATASETS:
        d = fd[fd["dataset"] == ds].copy()
        d["true_label"] = d["true_label"].astype(str)
        d["pred_label"] = d["pred_label"].astype(str)
        models = [m for m in MODELS if m in d["model"].unique()]
        if not models:
            continue
        classes = sorted(set(d["true_label"]) | set(d["pred_label"]))
        idx = {c: i for i, c in enumerate(classes)}
        n = len(models)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), dpi=130,
                                 squeeze=False)
        for k, m in enumerate(models):
            ax = axes[k // cols][k % cols]
            dm = d[d["model"] == m]
            cm = np.zeros((len(classes), len(classes)))
            for t, p in zip(dm["true_label"], dm["pred_label"]):
                cm[idx[t], idx[p]] += 1
            cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(PRETTY.get(m, m), fontsize=9)
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=90, fontsize=6)
            ax.set_yticklabels(classes, fontsize=6)
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(f"Row-normalised confusion matrices — {ds}", y=1.0)
        _save(fig, f"fig03_confusion_{ds}.png")


def experiment_A(fm):
    """Table 4 + Figures 5/6 — performance vs number of patients."""
    d = fm[fm["axis"] == "npatients"]
    if d.empty:
        return
    a, _ = agg(d, ["dataset", "model", "n_patients"])
    t = pretty_table(a, ["dataset", "model", "n_patients"],
                     PRIMARY + ["accuracy", "mcc", "macro_roc_auc_ovr"])
    t.sort_values(["dataset", "model", "n_patients"]).to_csv(
        os.path.join(TAB, "table04_patient_count_sweep.csv"), index=False)
    print("  table04_patient_count_sweep")
    for ds in DATASETS:
        for metric in PRIMARY:
            line_vs_axis(d[d["dataset"] == ds], "n_patients", metric,
                         f"{metric.replace('_', ' ')} vs #patients — {ds}",
                         f"fig05_npatients_{metric}_{ds}.png", "number of patients")


def experiment_B(fm):
    """Tables 5/6 + Figures 8/10/11 — number & combination of omics."""
    d = fm[fm["axis"] == "nomics"]
    if d.empty:
        return
    a, _ = agg(d, ["dataset", "model", "n_omics", "omic_combination"])
    # Table 6 — full leaderboard
    lb = pretty_table(a, ["dataset", "model", "n_omics", "omic_combination"],
                      PRIMARY + ["accuracy", "mcc"])
    lb.sort_values(["dataset", "model", "n_omics"]).to_csv(
        os.path.join(TAB, "table06_omic_combination_leaderboard.csv"), index=False)
    print("  table06_omic_combination_leaderboard")
    # Table 5 — best combination at each number of omics (by balanced accuracy)
    best = (a.sort_values("balanced_accuracy_mean", ascending=False)
              .groupby(["dataset", "model", "n_omics"], as_index=False).first())
    b = pretty_table(best, ["dataset", "model", "n_omics", "omic_combination"],
                     PRIMARY + ["accuracy"])
    b.sort_values(["dataset", "model", "n_omics"]).to_csv(
        os.path.join(TAB, "table05_best_combination_per_nomics.csv"), index=False)
    print("  table05_best_combination_per_nomics")
    # Figure 8 — best performance vs number of omics
    for ds in DATASETS:
        for metric in PRIMARY:
            sub = best[best["dataset"] == ds]
            if sub.empty:
                continue
            fig, ax = _newfig()
            for m in MODELS:
                md = sub[sub["model"] == m].sort_values("n_omics")
                if md.empty:
                    continue
                ax.plot(md["n_omics"], md[f"{metric}_mean"], "-o", ms=4,
                        color=COLOR[m], label=PRETTY[m])
                ax.fill_between(md["n_omics"],
                                md[f"{metric}_mean"] - md[f"{metric}_sd"].fillna(0),
                                md[f"{metric}_mean"] + md[f"{metric}_sd"].fillna(0),
                                color=COLOR[m], alpha=0.15)
            ax.set_xlabel("number of omics (best combination)")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"Best {metric.replace('_', ' ')} vs #omics — {ds}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
            _save(fig, f"fig08_best_nomics_{metric}_{ds}.png")


def experiment_C(fm):
    """Tables 8/10 + Figures 12/15 — omic missingness."""
    d = fm[fm["axis"] == "missing"]
    if d.empty:
        return
    a, _ = agg(d, ["dataset", "model", "include_non_intersection_frac"])
    t = pretty_table(a, ["dataset", "model", "include_non_intersection_frac"],
                     PRIMARY + ["accuracy", "mcc", "macro_roc_auc_ovr"])
    t.sort_values(["dataset", "model", "include_non_intersection_frac"]).to_csv(
        os.path.join(TAB, "table08_missingness_results.csv"), index=False)
    print("  table08_missingness_results")
    # Table 10 — robustness ranking: slope of balanced accuracy vs frac + drop from best
    rows = []
    for (ds, m), g in a.groupby(["dataset", "model"]):
        g = g.sort_values("include_non_intersection_frac")
        x = g["include_non_intersection_frac"].astype(float).values
        y = g["balanced_accuracy_mean"].values
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else np.nan
        rows.append({"dataset": ds, "model": m, "bacc_at_full": y[np.argmax(x)],
                     "bacc_at_least": y[np.argmin(x)], "drop": y[np.argmax(x)] - y[np.argmin(x)],
                     "slope_bacc_per_frac": slope})
    rob = pd.DataFrame(rows).sort_values(["dataset", "drop"])
    rob.round(4).to_csv(os.path.join(TAB, "table10_missingness_robustness.csv"), index=False)
    print("  table10_missingness_robustness")
    for ds in DATASETS:
        for metric in PRIMARY:
            line_vs_axis(d[d["dataset"] == ds], "include_non_intersection_frac", metric,
                         f"{metric.replace('_', ' ')} vs omic overlap frac — {ds}",
                         f"fig12_missing_{metric}_{ds}.png", "include_non_intersection_frac")


def experiment_D(fm):
    """Tables 11/13 + Figures 16 — percentage / count of features per omic."""
    d = fm[fm["axis"] == "ratio"]
    if d.empty:
        return
    xcol = "ratio_per_omic" if d["ratio_per_omic"].notna().any() else "k_per_omic"
    a, _ = agg(d, ["dataset", "model", xcol])
    t = pretty_table(a, ["dataset", "model", xcol], PRIMARY + ["accuracy", "mcc"])
    t.sort_values(["dataset", "model", xcol]).to_csv(
        os.path.join(TAB, "table11_feature_sweep.csv"), index=False)
    print("  table11_feature_sweep")
    # Table 13 — feature-efficiency: smallest budget within 1 SD of each model's best bacc
    rows = []
    for (ds, m), g in a.groupby(["dataset", "model"]):
        g = g.sort_values(xcol)
        best = g["balanced_accuracy_mean"].max()
        bsd = g.loc[g["balanced_accuracy_mean"].idxmax(), "balanced_accuracy_sd"]
        thr = best - (0 if pd.isna(bsd) else bsd)
        elig = g[g["balanced_accuracy_mean"] >= thr]
        pick = elig.iloc[0]
        rows.append({"dataset": ds, "model": m, "efficient_budget": pick[xcol],
                     "bacc_at_budget": round(pick["balanced_accuracy_mean"], 4),
                     "best_bacc": round(best, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "table13_feature_efficiency.csv"), index=False)
    print("  table13_feature_efficiency")
    for ds in DATASETS:
        for metric in PRIMARY:
            line_vs_axis(d[d["dataset"] == ds], xcol, metric,
                         f"{metric.replace('_', ' ')} vs feature budget — {ds}",
                         f"fig16_features_{metric}_{ds}.png", xcol, logx=True)


def cross_dataset(fm):
    """Tables 14/15 + Figure 20 — cross-dataset method summary and ranks."""
    fd = full_data_mask(fm)
    if fd.empty:
        return
    a, _ = agg(fd, ["dataset", "model"])
    # rank per dataset by balanced accuracy (1 = best)
    a["rank"] = a.groupby("dataset")["balanced_accuracy_mean"].rank(ascending=False)
    summ = a.groupby("model").agg(
        mean_balanced_accuracy=("balanced_accuracy_mean", "mean"),
        mean_macro_f1=("macro_f1_mean", "mean"),
        mean_rank=("rank", "mean"),
        n_datasets=("dataset", "nunique")).reset_index().sort_values("mean_rank")
    summ.round(4).to_csv(os.path.join(TAB, "table14_cross_dataset_summary.csv"), index=False)
    print("  table14_cross_dataset_summary")
    # Table 15 — winner per dataset
    win = (a.sort_values("balanced_accuracy_mean", ascending=False)
             .groupby("dataset", as_index=False).first()[
                 ["dataset", "model", "balanced_accuracy_mean", "macro_f1_mean"]])
    win.round(4).to_csv(os.path.join(TAB, "table15_dataset_winners.csv"), index=False)
    print("  table15_dataset_winners")
    # Figure 20 — method rank across datasets
    piv = a.pivot_table(index="model", columns="dataset", values="rank").reindex(
        [m for m in MODELS if m in a["model"].unique()])
    fig, ax = _newfig(1.6 * len(DATASETS) + 3, 0.6 * piv.shape[0] + 2)
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([PRETTY.get(m, m) for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9)
    ax.set_title("Method rank by balanced accuracy (1 = best)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="rank")
    _save(fig, "fig20_method_rank.png")


def cost_figure(fm):
    """Figure 22 — performance vs computational cost (full-data)."""
    fd = full_data_mask(fm)
    if fd.empty or "runtime_seconds" not in fd:
        return
    per_seed = fd.groupby(["dataset", "model", "seed"]).agg(
        bacc=("balanced_accuracy", "mean"), rt=("runtime_seconds", "mean")).reset_index()
    g = per_seed.groupby(["dataset", "model"]).agg(
        bacc=("bacc", "mean"), rt=("rt", "mean")).reset_index()
    fig, ax = _newfig()
    for m in MODELS:
        md = g[g["model"] == m]
        if md.empty:
            continue
        ax.scatter(md["rt"], md["bacc"], color=COLOR[m], label=PRETTY[m], s=45)
    ax.set_xscale("log")
    ax.set_xlabel("runtime per fold (s, log)")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Performance vs computational cost (full-data)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig22_performance_vs_cost.png")


def stats_table(fm):
    """Table 16 — pairwise method comparisons at the full-data point.

    Per §11: repeat-level (seed) differences in balanced accuracy & macro F1,
    paired within dataset+seed; bootstrap 95% CI, Wilcoxon signed-rank, BH-FDR.
    """
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
    # Benjamini-Hochberg FDR within each metric
    t["p_fdr"] = np.nan
    for metric in PRIMARY:
        mask = t["metric"] == metric
        p = t.loc[mask, "wilcoxon_p"].values
        order = np.argsort(np.where(np.isnan(p), 1.0, p))
        n = len(p)
        adj = np.full(n, np.nan)
        prev = 1.0
        for rank, i in enumerate(reversed(order), start=1):
            k = n - rank + 1
            if np.isnan(p[i]):
                continue
            prev = min(prev, p[i] * n / k)
            adj[i] = prev
        t.loc[mask, "p_fdr"] = adj
    t.round(4).to_csv(os.path.join(TAB, "table16_pairwise_comparisons.csv"), index=False)
    print("  table16_pairwise_comparisons")


def main():
    print("loading canonical shards ...")
    fm = _load("fold_metrics")
    pc = _load("per_class")
    pred = _load("predictions")
    if fm.empty:
        print("no fold_metrics shards found; nothing to aggregate yet.")
        return
    print(f"fold_metrics: {len(fm)} rows, "
          f"{fm[['dataset','model','axis']].drop_duplicates().shape[0]} cells")
    print("writing tables and figures ...")
    table2_main(fm)
    table3_perclass(pc)
    confusion_figs(pred)
    experiment_A(fm)
    experiment_B(fm)
    experiment_C(fm)
    experiment_D(fm)
    cross_dataset(fm)
    cost_figure(fm)
    stats_table(fm)
    # manifest of what was produced
    tabs = sorted(os.path.basename(x) for x in glob.glob(os.path.join(TAB, "*.csv")))
    figs = sorted(os.path.basename(x) for x in glob.glob(os.path.join(FIG, "*.png")))
    with open(os.path.join(HERE, "STAGE2_MANIFEST.json"), "w") as f:
        json.dump({"tables": tabs, "figures": figs,
                   "cells_present": fm[["dataset", "model", "axis"]].drop_duplicates()
                   .to_dict("records")}, f, indent=2, default=str)
    print(f"done: {len(tabs)} tables, {len(figs)} figures -> report/tables, report/figures")


if __name__ == "__main__":
    main()
