"""Full classification metric battery per the benchmark evaluation README.

Given per-fold reconstructed (y_true, y_pred, proba, classes), compute the
fold-level metrics (primary + secondary + probabilistic) and per-class metrics.
Probability-based metrics are skipped (NaN) when proba is unavailable (e.g. SVM).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def _onehot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes.tolist())}
    Y = np.zeros((len(y), len(classes)), dtype=float)
    for r, v in enumerate(y):
        Y[r, idx[v]] = 1.0
    return Y


def _ece(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(axis=1)
    pred = classes[proba.argmax(axis=1)]
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fold_metrics(y_true, y_pred, proba, classes) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    out: Dict[str, float] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    # probability-based
    prob_keys = ["macro_roc_auc_ovr", "weighted_roc_auc_ovr", "macro_pr_auc_ovr",
                 "log_loss", "brier_score", "ece"]
    if proba is None or np.isnan(np.asarray(proba)).any():
        for k in prob_keys:
            out[k] = float("nan")
        return out
    proba = np.asarray(proba, dtype=float)
    Y = _onehot(y_true, classes)
    present = Y.sum(axis=0) > 0  # classes present in this test fold
    try:
        out["macro_roc_auc_ovr"] = float(
            roc_auc_score(Y[:, present], proba[:, present], average="macro", multi_class="ovr")
        )
        out["weighted_roc_auc_ovr"] = float(
            roc_auc_score(Y[:, present], proba[:, present], average="weighted", multi_class="ovr")
        )
    except Exception:
        out["macro_roc_auc_ovr"] = float("nan")
        out["weighted_roc_auc_ovr"] = float("nan")
    try:
        out["macro_pr_auc_ovr"] = float(
            average_precision_score(Y[:, present], proba[:, present], average="macro")
        )
    except Exception:
        out["macro_pr_auc_ovr"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=classes))
    except Exception:
        out["log_loss"] = float("nan")
    try:
        out["brier_score"] = float(np.mean(np.sum((proba - Y) ** 2, axis=1)))
    except Exception:
        out["brier_score"] = float("nan")
    out["ece"] = _ece(y_true, proba, classes)
    return out


def per_class_metrics(y_true, y_pred, proba, classes) -> List[Dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    Y = _onehot(y_true, classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    rows = []
    for i, c in enumerate(classes.tolist()):
        support = int((y_true == c).sum())
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        roc = pr = float("nan")
        if proba is not None and not np.isnan(np.asarray(proba)).any() and 0 < support < len(y_true):
            try:
                roc = float(roc_auc_score(Y[:, i], np.asarray(proba)[:, i]))
                pr = float(average_precision_score(Y[:, i], np.asarray(proba)[:, i]))
            except Exception:
                pass
        rows.append({
            "class": c, "support": support, "prevalence": support / len(y_true),
            "precision": float(prec), "recall": float(rec), "specificity": float(spec),
            "f1": float(f1), "roc_auc_ovr": roc, "pr_auc_ovr": pr,
            "predicted_frequency": int((y_pred == c).sum()),
        })
    return rows
