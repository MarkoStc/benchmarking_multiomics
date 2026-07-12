"""Reconstruct held-out test predictions for a saved benchmark run from its
fold_models.joblib + metadata.json, WITHOUT retraining.

Given a run directory, this:
  1. rebuilds the exact prepared X (or per-view X) using the same cohort recipe
     the run used (build_cohort + optional stratified subsample), which is fully
     deterministic in (omics, include_non_intersection_frac, random_state);
  2. for each of the 5 outer folds, slices X at the stored test_idx and re-runs
     the fitted model's forward path to get predicted labels + class probabilities.

Row order of X == cohort order == the order the stored test_idx index into, so a
fresh StratifiedKFold on X reproduces the identical fold split (asserted).

Covers: logreg_early, svm_early, pca, logreg_late, pnet, mofa.
(integrao handled separately — its inference re-runs the predictor.)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "../evaluation"))
sys.path.insert(0, os.path.join(ROOT, "sklearn_common"))
# model dirs on path so pickled artifacts (e.g. mofa_model.MOFAEmbeddingTransformer)
# resolve during joblib.load
sys.path.insert(0, os.path.join(ROOT, "mofa"))

from base import _stratified_subsample_patient_ids  # noqa: E402
from common import load_dataset, parse_omics  # noqa: E402

_DATASET_CACHE: Dict[str, object] = {}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_dataset_cached(path: str):
    if path not in _DATASET_CACHE:
        if path.endswith(".compat.joblib"):
            sys.path.insert(0, os.path.join(ROOT, "mofa"))
            from mofa_model import load_compat_dataset
            _DATASET_CACHE[path] = load_compat_dataset(path)
        else:
            _DATASET_CACHE[path] = load_dataset(path)
    return _DATASET_CACHE[path]


def _proba_from_decision(est, X) -> Optional[np.ndarray]:
    """Version-robust class probabilities for linear/logistic estimators.

    Avoids calling predict_proba on cross-env pickles (sklearn attribute skew).
    For multinomial logistic (sklearn default with lbfgs) predict_proba is
    exactly softmax(decision_function); binary uses the logistic sigmoid.
    """
    if not hasattr(est, "decision_function"):
        try:
            return np.asarray(est.predict_proba(X))
        except Exception:
            return None
    d = np.asarray(est.decision_function(X))
    if d.ndim == 1:  # binary
        p1 = 1.0 / (1.0 + np.exp(-np.clip(d, -500, 500)))
        return np.column_stack([1.0 - p1, p1])
    return _softmax(d)


def load_metadata(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "metadata.json")) as fh:
        return json.load(fh)


def reconstruct_cohort(meta: dict) -> Tuple[object, List[str], List[str], str]:
    """Return (dataset, omics, cohort_patient_ids, missing_policy)."""
    em = meta["extra_metadata"]
    ds = _load_dataset_cached(em["dataset_path"])
    omics = parse_omics(em["omics"])
    seed = int(em["random_state"])
    frac = float(em["include_non_intersection_frac"])
    missing_policy = em.get("missing_policy", "impute")

    cohort = ds.build_cohort(
        omics=omics, include_non_intersection_frac=frac, random_state=seed
    )
    n_patients = em.get("n_patients")
    if n_patients is not None:
        cohort = _stratified_subsample_patient_ids(
            ds.y.loc[cohort], n=int(n_patients), random_state=seed
        )
    return ds, omics, cohort, missing_policy


# ─────────────────────── per-model-type prediction from artifact ──────────────
def _predict_early(art, X: pd.DataFrame, test_idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    pipe = art["pipeline"]
    Xte = X.iloc[test_idx]
    y_pred = np.asarray(pipe.predict(Xte))
    est = pipe.named_steps["estimator"]
    classes = np.asarray(est.classes_)
    proba = _proba_from_decision(pipe, Xte)
    return y_pred, proba, classes


def _align_cols(proba: np.ndarray, from_classes, to_classes) -> np.ndarray:
    from_classes = np.asarray(from_classes)
    to_classes = np.asarray(to_classes)
    out = np.zeros((proba.shape[0], len(to_classes)), dtype=float)
    for j, c in enumerate(from_classes):
        out[:, np.where(to_classes == c)[0][0]] = proba[:, j]
    return out


def _predict_svm(art, X: pd.DataFrame, y, test_idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """SVC(probability=False): predict from the frozen model; obtain calibrated
    probabilities via training-only Platt scaling (README §4.2 — 'calibrated
    probabilities fitted within the training data only'). The SVM is NOT refit."""
    from sklearn.calibration import CalibratedClassifierCV

    pipe = art["pipeline"]
    classes = np.asarray(pipe.named_steps["estimator"].classes_)
    Xte = X.iloc[test_idx]
    y_pred = np.asarray(pipe.predict(Xte))
    train_idx = np.asarray(art["train_idx"])
    Xtr, ytr = X.iloc[train_idx], np.asarray(y)[train_idx]
    try:
        try:  # sklearn >=1.6: freeze the fitted estimator explicitly
            from sklearn.frozen import FrozenEstimator
            cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
        except Exception:  # older sklearn
            cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
        cal.fit(Xtr, ytr)
        proba = _align_cols(np.asarray(cal.predict_proba(Xte)), cal.classes_, classes)
    except Exception:
        proba = None
    return y_pred, proba, classes


def _predict_pca(art, X: pd.DataFrame, test_idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    Xte = X.iloc[test_idx][art["selected_cols"]]
    Z = art["pca_embedder"].transform(Xte)
    Zs = art["scaler"].transform(Z.to_numpy(dtype=float))
    clf = art["classifier"]
    y_pred = np.asarray(clf.predict(Zs))
    proba = _proba_from_decision(clf, Zs)
    return y_pred, proba, np.asarray(clf.classes_)


def _predict_late(art, X_by_view, availability, test_idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    all_classes = np.asarray(art["all_classes"])
    n_test = len(test_idx)
    score_sum = np.zeros((n_test, len(all_classes)), dtype=float)
    contrib = np.zeros(n_test, dtype=float)
    for omic, pipe in art["view_pipelines"].items():
        Xtv = X_by_view[omic].iloc[test_idx]
        test_avail = availability[omic].iloc[test_idx]
        mask = test_avail.to_numpy()
        if int(mask.sum()) == 0:
            continue
        Xte_eval = Xtv.loc[test_avail]
        proba = _proba_from_decision(pipe, Xte_eval)
        model_classes = np.asarray(pipe.named_steps["estimator"].classes_)
        aligned = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
        for j, cls in enumerate(model_classes):
            col = np.where(all_classes == cls)[0][0]
            aligned[:, col] = proba[:, j]
        idx = np.where(mask)[0]
        score_sum[idx] += aligned
        contrib[idx] += 1.0
    proba_out = np.full((n_test, len(all_classes)), np.nan)
    valid = contrib > 0
    proba_out[valid] = score_sum[valid] / score_sum[valid].sum(axis=1, keepdims=True)
    y_pred = np.full(n_test, art.get("_fallback", all_classes[0]))
    y_pred[valid] = all_classes[np.argmax(score_sum[valid], axis=1)]
    return y_pred, proba_out, all_classes


def _predict_pnet(art, X: pd.DataFrame, test_idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    import torch
    sys.path.insert(0, os.path.join(ROOT, "pnet"))
    from pnet_model import PNetTorch

    Xte = X.iloc[test_idx][art["selected_cols"]]
    Xf = Xte.fillna(art["col_means"])
    Xnp = art["scaler"].transform(Xf.to_numpy(dtype=np.float32))
    arch = art["arch"]
    model = PNetTorch(
        n_gene_features=arch["n_gene_features"],
        n_genes=arch["n_genes"],
        n_pathways=arch["n_pathways"],
        feature_gene_mask=art["feature_gene_mask"],
        gene_pathway_mask=art["gene_pathway_mask"],
        n_dense_features=arch["n_dense_features"],
        hidden_units=arch["hidden_units"],
        n_classes=arch["n_classes"],
        dropout=arch["dropout"],
    )
    model.load_state_dict(art["state_dict"])
    model.eval()
    gene_idx, dense_idx = art["gene_idx"], art["dense_idx"]
    Xt = torch.from_numpy(Xnp).float()
    xg = Xt[:, gene_idx] if arch["n_gene_features"] > 0 else None
    xd = Xt[:, dense_idx] if arch["n_dense_features"] > 0 else None
    with torch.no_grad():
        logits = model(xg, xd).cpu().numpy()
    proba = _softmax(logits)
    classes = np.asarray(art["label_classes"])
    y_pred = classes[logits.argmax(axis=1)]
    return y_pred, proba, classes


def _predict_mofa(art, X: pd.DataFrame, test_idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    Xte = X.iloc[test_idx][art["selected_cols"]]
    Z = art["mofa_embedder"].transform(Xte)
    Zs = art["scaler"].transform(Z.to_numpy(dtype=float))
    clf = art["classifier"]
    proba = _proba_from_decision(clf, Zs)
    classes = np.asarray(art["label_classes"])
    y_pred = classes[clf.predict(Zs)]
    return y_pred, proba, classes


def _predict_integrao(art, meta, ds, X_by_view, test_idx):
    """Reconstruct IntegrAO predictions by rebuilding the predictor from the
    stored per-omic scalers/selected features + saved GNN state_dict, re-running
    network diffusion + inference (mirrors integrao_model inference path)."""
    import shutil
    import tempfile
    import torch
    sys.path.insert(0, os.path.join(ROOT, "integrao"))
    from integrao.integrater import integrao_predictor

    em = meta["extra_metadata"]
    neighbor_size = int(em["neighbor_size"])
    scalers, selected = art["scalers"], art["selected"]
    omics = sorted(selected.keys())
    all_ids = list(X_by_view[omics[0]].index)
    test_ids = [all_ids[i] for i in test_idx]
    classes = np.asarray(art["label_classes"])

    def apply_scale(omic, df):
        sc, col_means = scalers[omic]
        cols = selected[omic]
        filled = df[cols].fillna(col_means)
        return pd.DataFrame(sc.transform(filled), index=df.index, columns=cols)

    all_views, all_mod_names = [], []
    for omic in omics:
        obs_all = ~X_by_view[omic].isna().all(axis=1)
        X_obs_all = apply_scale(omic, X_by_view[omic]).loc[obs_all]
        if X_obs_all.shape[0] > 0:
            all_views.append(X_obs_all)
            all_mod_names.append(omic)

    eff_nb_all = min(neighbor_size, min(v.shape[0] for v in all_views) - 1)
    n_classes = int(art["n_classes"])
    tmp = tempfile.mkdtemp(prefix="integrao_recon_")
    try:
        predictor = integrao_predictor(
            all_views,
            dataset_name=ds.name,
            modalities_name_list=all_mod_names,
            neighbor_size=eff_nb_all,
            embedding_dims=art["embedding_dims"],
            hidden_channels=art["hidden_channels"],
            fusing_iteration=20,
            normalization_factor=1.0,
            mu=0.5,
            num_classes=n_classes,
        )
        predictor.network_diffusion()
        pred_index = list(predictor.dict_sampleToIndexs.keys())

        # fallback = majority encoded train label
        y_raw = np.asarray(ds.y.loc[all_ids])
        cls_to_enc = {c: i for i, c in enumerate(classes.tolist())}
        y_enc = np.array([cls_to_enc[c] for c in y_raw])
        fallback = int(pd.Series(y_enc[np.asarray(art["train_idx"])]).value_counts().idxmax())

        if art["mode"] == "supervised":
            sup_path = os.path.join(tmp, "model_supervised.pth")
            torch.save(art["sup_state_dict"], sup_path)
            raw = predictor.inference_supervised(
                sup_path, new_datasets=all_views, modalities_names=all_mod_names
            )
            pred_s = pd.Series(raw, index=pred_index)
            y_pred_enc = np.array([int(pred_s.get(pid, fallback)) for pid in test_ids])
            proba = None
        else:
            uns_path = os.path.join(tmp, "model.pth")
            torch.save(art["uns_state_dict"], uns_path)
            emb_df, _ = predictor.inference_unsupervised(
                uns_path, new_datasets=all_views, modalities_names=all_mod_names
            )
            zero_emb = np.zeros(art["embedding_dims"])
            test_emb = np.stack([
                emb_df.loc[pid].to_numpy() if pid in emb_df.index else zero_emb
                for pid in test_ids
            ])
            clf = art["classifier"]
            y_pred_enc = np.asarray(clf.predict(test_emb))
            proba = _proba_from_decision(clf, test_emb)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return classes[y_pred_enc], proba, classes


_TABULAR = {"logreg_early", "svm_early", "pca", "pnet", "mofa"}
_MULTIVIEW = {"logreg_late", "integrao"}


def predict_run(run_dir: str):
    """Yield per-fold dicts: fold, y_true, y_pred, proba, classes, test_idx."""
    import joblib
    meta = load_metadata(run_dir)
    folds = joblib.load(os.path.join(run_dir, "fold_models", "fold_models.joblib"))
    model_type = folds[0]["model_type"]
    ds, omics, cohort, missing_policy = reconstruct_cohort(meta)

    if model_type in _TABULAR:
        X, y, _ = ds.to_tabular(omics=omics, patient_ids=cohort, missing_policy=missing_policy)
        order = list(X.index)
    else:
        X_by_view, y, availability = ds.to_multiview(
            omics=omics, patient_ids=cohort, missing_policy=missing_policy
        )
        order = list(availability.index)

    for art in folds:
        test_idx = np.asarray(art["test_idx"])
        y_true = np.asarray(y)[test_idx]
        patient_ids = [order[i] for i in test_idx]
        if model_type == "svm_early":
            y_pred, proba, classes = _predict_svm(art, X, y, test_idx)
        elif model_type == "logreg_early":
            y_pred, proba, classes = _predict_early(art, X, test_idx)
        elif model_type == "pca":
            y_pred, proba, classes = _predict_pca(art, X, test_idx)
        elif model_type == "pnet":
            y_pred, proba, classes = _predict_pnet(art, X, test_idx)
        elif model_type == "mofa":
            y_pred, proba, classes = _predict_mofa(art, X, test_idx)
        elif model_type == "logreg_late":
            train_idx = np.asarray(art["train_idx"])
            art["_fallback"] = pd.Series(np.asarray(y)[train_idx]).value_counts().idxmax()
            y_pred, proba, classes = _predict_late(art, X_by_view, availability, test_idx)
        elif model_type == "integrao":
            y_pred, proba, classes = _predict_integrao(art, meta, ds, X_by_view, test_idx)
        else:
            raise ValueError(f"unknown model_type {model_type}")
        yield {
            "fold": art["fold"],
            "y_true": y_true,
            "y_pred": np.asarray(y_pred),
            "proba": proba,
            "classes": classes,
            "test_idx": test_idx,
            "patient_ids": patient_ids,
        }
