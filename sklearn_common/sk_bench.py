"""Sklearn-based benchmark models (early / intermediate / late integration).

Four models matching the benchmark figure, all built on the shared BaseModel
framework and all persisting the 5 outer-CV fitted models per run:

  - LogRegEarlyModel : early integration  (concat omics -> LogisticRegression)
  - SVMEarlyModel    : early integration  (concat omics -> SVC)
  - PCALogRegModel   : intermediate       (per-omic PCA -> concat -> LogisticRegression)
  - LogRegLateModel  : late integration   (per-omic LogReg -> soft-vote fusion)

Feature selection (ANOVA k/ratio), imputation, scaling, splits, seeds and the
nested-CV outer loop are identical to the other models. Each job runs one fixed
HP combination (the HP grid is swept across jobs by make_experiments.sh), exactly
like MOFA/IntegrAO/P-NET.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation"))
from base import (  # noqa: E402
    BaseModel,
    CVSpec,
    ExperimentResult,
    ExperimentSpec,
    FeatureSelectionSpec,
    FoldResult,
    FusionSpec,
    MultiOmicsDataset,
    MultiViewPreparedData,
    TabularPreparedData,
)


def dump_fold_artifacts(model, run_dir) -> None:
    """Persist per-fold fitted models collected during nested CV into run_dir."""
    import joblib
    from pathlib import Path

    fa = getattr(model, "fold_artifacts_", None)
    if not fa:
        return
    d = Path(run_dir) / "fold_models"
    d.mkdir(exist_ok=True)
    joblib.dump(fa, d / "fold_models.joblib", compress=3)


# ─────────────────────────────── EARLY integration ───────────────────────────
class _EarlyBase(BaseModel):
    """Concatenate all omics, ANOVA-select, scale, then a single classifier."""

    model_type = "early"

    def __init__(
        self,
        cv_spec: CVSpec = CVSpec(),
        feature_selection_spec: FeatureSelectionSpec = FeatureSelectionSpec(
            method="anova", k_per_omic=100
        ),
        n_jobs: int = 1,
    ):
        super().__init__(
            cv_spec=cv_spec,
            feature_selection_spec=feature_selection_spec,
            n_jobs=n_jobs,
            integration="early",
        )

    def hyperparameter_grid(self):
        return [{}]

    def _fold_best_params(self) -> Dict[str, Any]:
        return {}

    def prepare_experiment_data(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        patient_ids: Sequence[str],
        missing_policy: Literal["intersection", "impute"] = "impute",
    ) -> TabularPreparedData:
        X, y, feature_groups = dataset.to_tabular(
            omics=omics, patient_ids=patient_ids, missing_policy=missing_policy
        )
        return TabularPreparedData(
            patient_ids=list(X.index),
            y=y,
            n_features_input=int(X.shape[1]),
            X=X,
            feature_groups=feature_groups,
        )

    def _nested_cv_evaluate(
        self, data, experiment_spec: ExperimentSpec, random_state: int = 0
    ) -> ExperimentResult:
        if not isinstance(data, TabularPreparedData):
            raise TypeError(f"{self.__class__.__name__} requires TabularPreparedData.")
        X = data.X
        y = data.y
        feature_groups = data.feature_groups
        fs_spec = experiment_spec.feature_selection
        seed = experiment_spec.cv.random_state

        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits, shuffle=True, random_state=seed
        )

        folds: List[FoldResult] = []
        self.fold_artifacts_ = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

            pipe = self._build_pipeline(
                feature_groups=feature_groups, feature_selection_spec=fs_spec
            )
            t0 = time.perf_counter()
            pipe.fit(X_train, y_train)
            t_fit = time.perf_counter() - t0
            t1 = time.perf_counter()
            y_pred = pipe.predict(X_test)
            t_pred = time.perf_counter() - t1

            self.fold_artifacts_.append(
                {
                    "model_type": self.model_type,
                    "pipeline": pipe,
                    "best_params": self._fold_best_params(),
                    "fold": int(fold_idx),
                    "train_idx": [int(i) for i in train_idx],
                    "test_idx": [int(i) for i in test_idx],
                }
            )
            folds.append(
                FoldResult(
                    fold=fold_idx,
                    n_train=int(len(train_idx)),
                    n_test=int(len(test_idx)),
                    accuracy=float(accuracy_score(y_test, y_pred)),
                    balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                    fit_time_sec=float(t_fit),
                    predict_time_sec=float(t_pred),
                    total_time_sec=float(t_fit + t_pred),
                    best_params=self._fold_best_params(),
                )
            )

        return ExperimentResult(
            folds=folds,
            metadata={
                "experiment": asdict(experiment_spec),
                "n_samples": int(len(data.y)),
                "n_features_input": int(data.n_features_input),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "integration": self.model_type,
            },
        )


class LogRegEarlyModel(_EarlyBase):
    model_type = "logreg_early"

    def __init__(self, *args, downstream_c: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.downstream_c = float(downstream_c)

    def build_estimator(self):
        return LogisticRegression(
            C=self.downstream_c, max_iter=5000, class_weight="balanced"
        )

    def _fold_best_params(self):
        return {"downstream_c": self.downstream_c}


class SVMEarlyModel(_EarlyBase):
    model_type = "svm_early"

    def __init__(self, *args, svm_c: float = 1.0, kernel: str = "rbf", **kwargs):
        super().__init__(*args, **kwargs)
        self.svm_c = float(svm_c)
        self.kernel = str(kernel)

    def build_estimator(self):
        return SVC(
            C=self.svm_c,
            kernel=self.kernel,
            class_weight="balanced",
            probability=False,
        )

    def _fold_best_params(self):
        return {"svm_c": self.svm_c, "kernel": self.kernel}


# ─────────────────────────── INTERMEDIATE (PCA) integration ───────────────────
def _anova_select(X_obs: pd.DataFrame, y_obs: np.ndarray, k: int) -> List[str]:
    from sklearn.feature_selection import f_classif

    cols = X_obs.columns.tolist()
    k = min(k, len(cols))
    scores: List[float] = []
    for col in cols:
        x = X_obs[col].to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 3 or np.unique(y_obs[mask]).size < 2:
            scores.append(-np.inf)
            continue
        try:
            scores.append(float(f_classif(x[mask].reshape(-1, 1), y_obs[mask])[0][0]))
        except Exception:
            scores.append(-np.inf)
    top = np.argsort(scores)[::-1][:k]
    return [cols[i] for i in top]


class PCAEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Per-omic PCA embedding. Each omic is median-imputed, standardized, then
    reduced to (up to) n_components; the per-omic embeddings are concatenated."""

    def __init__(self, feature_groups: Dict[str, List[str]], n_components: int = 15,
                 random_state: int = 0):
        self.feature_groups = feature_groups
        self.n_components = int(n_components)
        self.random_state = int(random_state)

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "PCAEmbeddingTransformer":
        self.medians_: Dict[str, pd.Series] = {}
        self.scalers_: Dict[str, StandardScaler] = {}
        self.pcas_: Dict[str, PCA] = {}
        self.view_cols_: Dict[str, List[str]] = {}
        self.output_features_: List[str] = []
        for omic, cols in self.feature_groups.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue
            Xo = X[cols]
            med = Xo.median(axis=0)
            med = med.fillna(0.0)
            Xf = Xo.fillna(med).to_numpy(dtype=float)
            scaler = StandardScaler().fit(Xf)
            Xs = scaler.transform(Xf)
            nc = min(self.n_components, Xs.shape[1], max(1, Xs.shape[0] - 1))
            pca = PCA(n_components=nc, random_state=self.random_state).fit(Xs)
            self.medians_[omic] = med
            self.scalers_[omic] = scaler
            self.pcas_[omic] = pca
            self.view_cols_[omic] = cols
            self.output_features_.extend([f"{omic}__pc{j}" for j in range(nc)])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for omic, cols in self.view_cols_.items():
            Xo = X[cols].fillna(self.medians_[omic]).to_numpy(dtype=float)
            Xs = self.scalers_[omic].transform(Xo)
            parts.append(self.pcas_[omic].transform(Xs))
        Z = np.hstack(parts) if parts else np.zeros((len(X), 0))
        return pd.DataFrame(Z, index=X.index, columns=self.output_features_)


class PCALogRegModel(BaseModel):
    model_type = "pca"

    def __init__(
        self,
        cv_spec: CVSpec = CVSpec(),
        feature_selection_spec: FeatureSelectionSpec = FeatureSelectionSpec(
            method="anova", k_per_omic=100
        ),
        n_components: int = 15,
        downstream_c: float = 1.0,
        n_jobs: int = 1,
    ):
        super().__init__(
            cv_spec=cv_spec,
            feature_selection_spec=feature_selection_spec,
            n_jobs=n_jobs,
            integration="early",
        )
        self.n_components = int(n_components)
        self.downstream_c = float(downstream_c)

    def build_estimator(self):
        raise NotImplementedError("PCALogRegModel uses its own training loop.")

    def hyperparameter_grid(self):
        return [{}]

    def prepare_experiment_data(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        patient_ids: Sequence[str],
        missing_policy: Literal["intersection", "impute"] = "impute",
    ) -> TabularPreparedData:
        X, y, feature_groups = dataset.to_tabular(
            omics=omics, patient_ids=patient_ids, missing_policy=missing_policy
        )
        return TabularPreparedData(
            patient_ids=list(X.index),
            y=y,
            n_features_input=int(X.shape[1]),
            X=X,
            feature_groups=feature_groups,
        )

    def _select_groups(self, X_train_df, y_train, fs_spec, feature_groups):
        if fs_spec.k_per_omic is not None:
            use_ratio, k_base = False, int(fs_spec.k_per_omic)
        elif fs_spec.ratio_per_omic is not None:
            use_ratio, ratio = True, float(fs_spec.ratio_per_omic)
        else:
            use_ratio, k_base = False, 200
        selected_groups: Dict[str, List[str]] = {}
        for omic, cols in feature_groups.items():
            cols = [c for c in cols if c in X_train_df.columns]
            if not cols:
                continue
            X_tr_omic = X_train_df[cols]
            obs = ~X_tr_omic.isna().all(axis=1)
            X_obs = X_tr_omic.loc[obs]
            y_obs = y_train[obs.to_numpy()]
            k = max(1, int(ratio * len(cols))) if use_ratio else k_base
            k = min(k, len(cols))
            if X_obs.shape[0] < 5 or np.unique(y_obs).size < 2:
                selected_groups[omic] = cols[:k]
            else:
                selected_groups[omic] = _anova_select(X_obs, y_obs, k)
        return selected_groups

    def _run_single_fold(self, X_train_df, y_train, X_test_df, y_test, fs_spec,
                         feature_groups, random_state):
        selected_groups = self._select_groups(X_train_df, y_train, fs_spec, feature_groups)
        selected_cols = [c for cols in selected_groups.values() for c in cols]
        if not selected_cols:
            selected_groups = {k: list(v) for k, v in feature_groups.items()}
            selected_cols = [c for cols in selected_groups.values() for c in cols]

        X_train_sel = X_train_df[selected_cols]
        X_test_sel = X_test_df[selected_cols]

        t0 = time.perf_counter()
        embed = PCAEmbeddingTransformer(
            feature_groups=selected_groups,
            n_components=self.n_components,
            random_state=random_state,
        )
        embed.fit(X_train_sel, y_train)
        Z_train = embed.transform(X_train_sel)
        Z_test = embed.transform(X_test_sel)

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train.to_numpy(dtype=float))
        Z_test_s = scaler.transform(Z_test.to_numpy(dtype=float))

        clf = LogisticRegression(C=self.downstream_c, max_iter=5000, class_weight="balanced")
        clf.fit(Z_train_s, y_train)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = clf.predict(Z_test_s)
        t_pred = time.perf_counter() - t1

        artifact = {
            "model_type": "pca",
            "pca_embedder": embed,
            "scaler": scaler,
            "classifier": clf,
            "selected_cols": list(selected_cols),
            "selected_groups": {k: list(v) for k, v in selected_groups.items()},
            "n_components": int(self.n_components),
        }
        return y_pred, t_fit, t_pred, artifact

    def _nested_cv_evaluate(self, data, experiment_spec: ExperimentSpec, random_state: int = 0):
        if not isinstance(data, TabularPreparedData):
            raise TypeError("PCALogRegModel requires TabularPreparedData.")
        X = data.X
        y = data.y
        feature_groups = data.feature_groups
        fs_spec = experiment_spec.feature_selection
        seed = experiment_spec.cv.random_state

        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits, shuffle=True, random_state=seed
        )
        folds: List[FoldResult] = []
        self.fold_artifacts_ = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train_df, X_test_df = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

            y_pred, t_fit, t_pred, artifact = self._run_single_fold(
                X_train_df, y_train, X_test_df, y_test, fs_spec, feature_groups, seed
            )
            artifact["fold"] = int(fold_idx)
            artifact["train_idx"] = [int(i) for i in train_idx]
            artifact["test_idx"] = [int(i) for i in test_idx]
            self.fold_artifacts_.append(artifact)

            folds.append(
                FoldResult(
                    fold=fold_idx,
                    n_train=int(len(train_idx)),
                    n_test=int(len(test_idx)),
                    accuracy=float(accuracy_score(y_test, y_pred)),
                    balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                    fit_time_sec=float(t_fit),
                    predict_time_sec=float(t_pred),
                    total_time_sec=float(t_fit + t_pred),
                    best_params={
                        "n_components": self.n_components,
                        "downstream_c": self.downstream_c,
                        "k_per_omic": fs_spec.k_per_omic,
                        "ratio_per_omic": fs_spec.ratio_per_omic,
                    },
                )
            )

        return ExperimentResult(
            folds=folds,
            metadata={
                "experiment": asdict(experiment_spec),
                "n_samples": int(len(data.y)),
                "n_features_input": int(data.n_features_input),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "integration": "pca",
            },
        )


# ─────────────────────────────── LATE integration ────────────────────────────
class LogRegLateModel(BaseModel):
    """Per-omic LogisticRegression; predictions fused by soft voting (predict_proba)."""

    model_type = "logreg_late"

    def __init__(
        self,
        cv_spec: CVSpec = CVSpec(),
        feature_selection_spec: FeatureSelectionSpec = FeatureSelectionSpec(
            method="anova", k_per_omic=100
        ),
        downstream_c: float = 1.0,
        n_jobs: int = 1,
    ):
        super().__init__(
            cv_spec=cv_spec,
            feature_selection_spec=feature_selection_spec,
            n_jobs=n_jobs,
            integration="late",
            fusion_spec=FusionSpec(voting="soft", score_method="predict_proba", on_missing_view="skip"),
        )
        self.downstream_c = float(downstream_c)

    def build_estimator(self):
        return LogisticRegression(C=self.downstream_c, max_iter=5000, class_weight="balanced")

    def hyperparameter_grid(self):
        return [{}]

    def prepare_experiment_data(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        patient_ids: Sequence[str],
        missing_policy: Literal["intersection", "impute"] = "impute",
    ) -> MultiViewPreparedData:
        X_by_view, y, availability = dataset.to_multiview(
            omics=omics, patient_ids=patient_ids, missing_policy=missing_policy
        )
        return MultiViewPreparedData(
            patient_ids=list(availability.index),
            y=y,
            n_features_input=int(sum(df.shape[1] for df in X_by_view.values())),
            X_by_view=X_by_view,
            availability=availability,
        )

    def _nested_cv_evaluate(self, data, experiment_spec: ExperimentSpec, random_state: int = 0):
        if not isinstance(data, MultiViewPreparedData):
            raise TypeError("LogRegLateModel requires MultiViewPreparedData.")
        y = data.y
        fs_spec = experiment_spec.feature_selection
        seed = experiment_spec.cv.random_state
        all_classes = np.unique(y)

        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits, shuffle=True, random_state=seed
        )

        folds: List[FoldResult] = []
        self.fold_artifacts_ = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(np.zeros(len(y)), y), start=1
        ):
            y_train, y_test = y[train_idx].copy(), y[test_idx].copy()
            fallback_class = pd.Series(y_train).value_counts().idxmax()
            n_test = len(test_idx)

            fitted_views: Dict[str, Any] = {}
            view_selected: Dict[str, List[str]] = {}
            t0 = time.perf_counter()
            for omic, X_view_full in data.X_by_view.items():
                X_train_view = X_view_full.iloc[train_idx].copy()
                train_avail = data.availability[omic].iloc[train_idx]
                if int(train_avail.sum()) == 0:
                    continue
                X_train_fit = X_train_view.loc[train_avail].copy()
                y_train_fit = y_train[train_avail.to_numpy()].copy()
                if np.unique(y_train_fit).size < 2:
                    continue
                pipe = self._build_pipeline(
                    feature_groups={omic: X_train_fit.columns.tolist()},
                    feature_selection_spec=fs_spec,
                )
                try:
                    pipe.fit(X_train_fit, y_train_fit)
                    fitted_views[omic] = pipe
                    sel = pipe.named_steps["selector"]
                    view_selected[omic] = list(getattr(sel, "selected_feature_names_", []))
                except Exception:
                    continue
            t_fit = time.perf_counter() - t0

            t1 = time.perf_counter()
            per_view_outputs: Dict[str, Dict[str, Any]] = {}
            for omic, pipe in fitted_views.items():
                X_test_view = data.X_by_view[omic].iloc[test_idx].copy()
                test_avail = data.availability[omic].iloc[test_idx]
                mask = test_avail.to_numpy()
                if int(mask.sum()) == 0:
                    continue
                X_test_eval = X_test_view.loc[test_avail].copy()
                try:
                    kind, output, model_classes = self._get_view_output(pipe, X_test_eval)
                    if kind == "soft":
                        output = self._align_scores(
                            scores=np.asarray(output),
                            model_classes=np.asarray(model_classes),
                            all_classes=all_classes,
                        )
                    per_view_outputs[omic] = {"mask": mask, "kind": kind, "output": output}
                except Exception:
                    continue

            y_pred = self._fuse_predictions(
                per_view_outputs=per_view_outputs,
                all_classes=all_classes,
                fallback_class=fallback_class,
                n_test=n_test,
            )
            t_pred = time.perf_counter() - t1

            self.fold_artifacts_.append(
                {
                    "model_type": "logreg_late",
                    "view_pipelines": fitted_views,
                    "view_selected": view_selected,
                    "downstream_c": self.downstream_c,
                    "fusion": asdict(self.fusion_spec),
                    "all_classes": [c for c in all_classes.tolist()],
                    "fold": int(fold_idx),
                    "train_idx": [int(i) for i in train_idx],
                    "test_idx": [int(i) for i in test_idx],
                }
            )
            folds.append(
                FoldResult(
                    fold=fold_idx,
                    n_train=int(len(train_idx)),
                    n_test=int(len(test_idx)),
                    accuracy=float(accuracy_score(y_test, y_pred)),
                    balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred)),
                    fit_time_sec=float(t_fit),
                    predict_time_sec=float(t_pred),
                    total_time_sec=float(t_fit + t_pred),
                    best_params={"downstream_c": self.downstream_c},
                )
            )

        return ExperimentResult(
            folds=folds,
            metadata={
                "experiment": asdict(experiment_spec),
                "n_samples": int(len(data.y)),
                "n_features_input": int(data.n_features_input),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "integration": "late",
            },
        )
