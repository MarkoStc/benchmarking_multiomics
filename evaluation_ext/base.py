#!/usr/bin/env python
# coding: utf-8

# # Unified, leakage-safe multi-omics benchmarking framework (TCGA-style)
# 
# This notebook provides a **production-quality, extensible benchmarking framework** for multi-omics **multiclass classification** in a TCGA-style setting (multiple omics “views”, missing-view patients, and multiple benchmark axes).
# 
# Key properties:
# 
# - **Object-oriented** design with a reusable `BaseModel` abstraction and model-specific subclasses.
# - A concrete, fully working example model: **`SVMModel`** (scikit-learn `SVC`) with nested CV hyperparameter tuning.
# - A `MultiOmicsDataset` abstraction that supports:
#   - multiple omics matrices (views),
#   - **view availability tracking**,
#   - selecting omics and patients,
#   - intersection-only cohorts or inclusion of incomplete patients (with leakage-safe imputation),
#   - missingness simulation (for demo).
# - Built-in benchmark axes (framework-level):
#   - **Accuracy vs Missingness**
#   - **Accuracy vs Computation Time**
#   - **Accuracy vs Number of Features**
#   - **Accuracy vs Number of Patients**
#   - **Accuracy vs Number of Modalities**
#   - **Accuracy vs Ratio of Selected Features**
# - Structured result objects (`dataclasses`) with per-fold metrics and metadata.
# - **Interpretability outputs** where applicable (linear SVM coefficients; documented fallback for non-linear kernels).
# 
# > Real **TCGA-BRCA** and **TCGA-GBM** matrices are not provided here, so the notebook generates **synthetic** multi-omics datasets with TCGA-like characteristics. It is obvious where real data would be plugged in.
# 

# In[1]:


from __future__ import annotations

import itertools
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RNG = np.random.default_rng(42)


# ## Result objects (structured, reproducible)
# 
# All benchmark methods return structured dataclasses:
# 
# - `FoldResult`: per-fold accuracy, balanced accuracy, runtime, best hyperparameters, and (optionally) interpretability summaries.
# - `ExperimentResult`: list of folds + aggregated mean/std + experiment metadata.
# - `GridResult`: mapping from a grid setting (e.g., `k=200`) to an `ExperimentResult`.
# 
# These objects are designed to be JSON-serializable via `.to_dict()` and easy to convert to pandas.
# 

# In[2]:


@dataclass
class InterpretabilityResult:
    """Interpretability outputs for a fitted model.

    For SVM:
      - linear kernel: exposes coefficients aligned to selected features (per class for OVR).
      - non-linear kernels: by default returns None (documented limitation).
    """
    supported: bool
    method: str
    feature_names: Optional[List[str]] = None
    coef: Optional[np.ndarray] = None  # shape: (n_classes, n_features) for multiclass OVR
    top_features_per_class: Optional[Dict[str, List[Tuple[str, float]]]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy arrays to lists for JSON-friendliness
        if d.get("coef") is not None:
            d["coef"] = np.asarray(d["coef"]).tolist()
        return d


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    fit_time_sec: float
    predict_time_sec: float
    total_time_sec: float
    best_params: Dict[str, Any]
    interpretability: Optional[InterpretabilityResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.interpretability is not None:
            d["interpretability"] = self.interpretability.to_dict()
        return d


@dataclass
class ExperimentResult:
    """An experiment result: outer-CV folds + aggregated statistics + metadata."""
    folds: List[FoldResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy_mean(self) -> float:
        return float(np.mean([f.accuracy for f in self.folds]))

    @property
    def accuracy_std(self) -> float:
        return float(np.std([f.accuracy for f in self.folds], ddof=1)) if len(self.folds) > 1 else 0.0

    @property
    def balanced_accuracy_mean(self) -> float:
        return float(np.mean([f.balanced_accuracy for f in self.folds]))

    @property
    def balanced_accuracy_std(self) -> float:
        return float(np.std([f.balanced_accuracy for f in self.folds], ddof=1)) if len(self.folds) > 1 else 0.0

    @property
    def total_time_mean_sec(self) -> float:
        return float(np.mean([f.total_time_sec for f in self.folds]))

    @property
    def total_time_std_sec(self) -> float:
        return float(np.std([f.total_time_sec for f in self.folds], ddof=1)) if len(self.folds) > 1 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folds": [f.to_dict() for f in self.folds],
            "summary": {
                "accuracy_mean": self.accuracy_mean,
                "accuracy_std": self.accuracy_std,
                "balanced_accuracy_mean": self.balanced_accuracy_mean,
                "balanced_accuracy_std": self.balanced_accuracy_std,
                "total_time_mean_sec": self.total_time_mean_sec,
                "total_time_std_sec": self.total_time_std_sec,
            },
            "metadata": self.metadata,
        }


@dataclass
class GridResult:
    """A mapping from a grid key (e.g., `k=200`) to an ExperimentResult."""
    results: Dict[Any, ExperimentResult]
    axis_name: str
    axis_values: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for k in self.axis_values:
            res = self.results[k]
            rows.append({
                self.axis_name: k,
                "balanced_accuracy_mean": res.balanced_accuracy_mean,
                "balanced_accuracy_std": res.balanced_accuracy_std,
                "accuracy_mean": res.accuracy_mean,
                "accuracy_std": res.accuracy_std,
                "total_time_mean_sec": res.total_time_mean_sec,
                "total_time_std_sec": res.total_time_std_sec,
                "n_folds": len(res.folds),
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis_name": self.axis_name,
            "axis_values": self.axis_values,
            "results": {str(k): v.to_dict() for k, v in self.results.items()},
            "metadata": self.metadata,
        }


# ## Dataset abstraction (`MultiOmicsDataset`)
# 
# `MultiOmicsDataset` wraps a TCGA-style multi-omics dataset:
# 
# - Stores views as `pd.DataFrame` in a dict, e.g. `{"mrna": X_mrna, "dnam": X_dnam, "rppa": X_rppa}`.
# - Tracks **per-patient view availability**.
# - Supports cohort construction:
#   - **intersection-only** patients (all selected views present),
#   - inclusion of a **fraction of non-intersection** patients (incomplete view availability).
# - Provides leakage-safe hooks for tabular models:
#   - concatenation of selected views into a single matrix,
#   - per-view feature groups to enable per-view feature selection.
# 
# In this demo, views contain NaNs for missing-view patients (block missingness), and the benchmark pipeline uses leakage-safe imputation inside CV folds.
# 

# In[3]:


@dataclass
class MultiOmicsDataset:
    """A lightweight multi-omics dataset wrapper (TCGA-style).

    Design choices:
    - Each view is a DataFrame with **the same patient index**.
    - Missing views are represented by rows that are entirely NaN for that view.
    - `availability` is computed per (patient, omic) as: row has at least one non-NaN.

    This makes it easy to:
    - select cohorts based on view availability,
    - concatenate views into a single tabular matrix for models like SVM / logistic regression,
    - keep feature names stable for interpretability.
    """
    name: Literal["TCGA-BRCA", "TCGA-GBM"]
    views: Dict[str, pd.DataFrame]  # keys are omic names
    y: pd.Series  # multiclass labels aligned with patient index
    patient_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Basic alignment checks
        if set(self.patient_ids) != set(self.y.index):
            raise ValueError("patient_ids must match y.index")
        for omic, df in self.views.items():
            if not df.index.equals(self.y.index):
                raise ValueError(f"View '{omic}' index must match y.index")
        # Precompute availability
        self._availability = self._compute_availability()

    @property
    def omics(self) -> List[str]:
        return sorted(self.views.keys())

    @property
    def availability(self) -> pd.DataFrame:
        """Boolean availability matrix: patients x omics."""
        return self._availability.copy()

    def _compute_availability(self) -> pd.DataFrame:
        avail = {}
        for omic, df in self.views.items():
            # Available if at least one feature is not NaN
            avail[omic] = ~df.isna().all(axis=1)
        return pd.DataFrame(avail, index=self.y.index)

    def summary(self, selected_omics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        omics = list(selected_omics) if selected_omics is not None else self.omics
        avail = self._availability[omics]
        per_omic_missing = (1.0 - avail.mean()).to_dict()
        intersection_rate = float(avail.all(axis=1).mean())
        return {
            "dataset": self.name,
            "n_patients": len(self.y),
            "n_classes": int(self.y.nunique()),
            "class_counts": self.y.value_counts().to_dict(),
            "omics": omics,
            "n_omics": len(omics),
            "features_per_omic": {o: int(self.views[o].shape[1]) for o in omics},
            "missing_rate_per_omic": per_omic_missing,
            "intersection_patient_rate": intersection_rate,
        }

    def select_omics(self, omics: Sequence[str]) -> "MultiOmicsDataset":
        omics = list(omics)
        missing = set(omics) - set(self.views.keys())
        if missing:
            raise KeyError(f"Unknown omics: {sorted(missing)}")
        new_views = {o: self.views[o].copy() for o in omics}
        return MultiOmicsDataset(
            name=self.name,
            views=new_views,
            y=self.y.copy(),
            patient_ids=list(self.patient_ids),
            metadata=dict(self.metadata),
        )

    def subset_patients(self, patient_ids: Sequence[str]) -> "MultiOmicsDataset":
        patient_ids = list(patient_ids)
        new_y = self.y.loc[patient_ids].copy()
        new_views = {o: df.loc[patient_ids].copy() for o, df in self.views.items()}
        return MultiOmicsDataset(
            name=self.name,
            views=new_views,
            y=new_y,
            patient_ids=patient_ids,
            metadata=dict(self.metadata),
        )

    def intersection_patients(self, omics: Sequence[str]) -> List[str]:
        omics = list(omics)
        avail = self._availability[omics]
        return avail.index[avail.all(axis=1)].tolist()

    def non_intersection_patients(self, omics: Sequence[str]) -> List[str]:
        omics = list(omics)
        avail = self._availability[omics]
        return avail.index[~avail.all(axis=1)].tolist()

    def build_cohort(
        self,
        omics: Sequence[str],
        include_non_intersection_frac: float = 0.0,
        random_state: int = 0,
    ) -> List[str]:
        """Return patient IDs for a cohort consisting of all intersection patients plus
        a fraction of non-intersection patients.

        - include_non_intersection_frac=0.0 => intersection-only
        - include_non_intersection_frac=1.0 => include all patients (for the selected omics)
        """
        if not (0.0 <= include_non_intersection_frac <= 1.0):
            raise ValueError("include_non_intersection_frac must be in [0, 1]")
        omics = list(omics)
        inter = self.intersection_patients(omics)
        non_inter = self.non_intersection_patients(omics)
        rng = np.random.default_rng(random_state)
        n_add = int(round(include_non_intersection_frac * len(non_inter)))
        add_ids = rng.choice(non_inter, size=n_add, replace=False).tolist() if n_add > 0 else []
        cohort = inter + add_ids
        return cohort

    def to_tabular(
        self,
        omics: Sequence[str],
        patient_ids: Optional[Sequence[str]] = None,
        missing_policy: Literal["intersection", "impute"] = "impute",
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[str]]]:
        """Concatenate selected omics into a single tabular matrix (patients x features).

        Returns:
          X_df: DataFrame with concatenated features, indexed by patient_id
          y: numpy array of labels aligned with X_df
          feature_groups: dict mapping omic -> list of column names belonging to that omic

        missing_policy:
          - 'intersection': drop patients missing any selected omic
          - 'impute': keep patients; missing views appear as NaNs (imputed leakage-safely in pipeline)
        """
        omics = list(omics)
        if patient_ids is None:
            patient_ids = list(self.patient_ids)
        else:
            patient_ids = list(patient_ids)

        if missing_policy == "intersection":
            patient_ids = [pid for pid in patient_ids if pid in self.intersection_patients(omics)]

        X_parts = []
        feature_groups: Dict[str, List[str]] = {}
        for omic in omics:
            df = self.views[omic].loc[patient_ids]
            X_parts.append(df)
            feature_groups[omic] = df.columns.tolist()

        X_df = pd.concat(X_parts, axis=1)
        y_arr = self.y.loc[patient_ids].to_numpy()
        return X_df, y_arr, feature_groups

    def to_multiview(
        self,
        omics: Sequence[str],
        patient_ids: Optional[Sequence[str]] = None,
        missing_policy: Literal["intersection", "impute"] = "impute",
    ) -> Tuple[Dict[str, pd.DataFrame], np.ndarray, pd.DataFrame]:
        omics = list(omics)

        if patient_ids is None:
            patient_ids = list(self.patient_ids)
        else:
            patient_ids = list(patient_ids)

        if missing_policy == "intersection":
            patient_ids = [pid for pid in patient_ids if pid in self.intersection_patients(omics)]

        X_by_view = {omic: self.views[omic].loc[patient_ids].copy() for omic in omics}
        y_arr = self.y.loc[patient_ids].to_numpy()
        availability = self._availability.loc[patient_ids, omics].copy()

        return X_by_view, y_arr, availability

    def with_additional_view_missingness(
        self,
        target_missing_rate_per_omic: float,
        omics: Optional[Sequence[str]] = None,
        random_state: int = 0,
    ) -> "MultiOmicsDataset":
        """Return a new dataset with *additional* block-wise view missingness applied
        (README 8.2 random block missingness). It only *adds* missing views (never
        unmasks); target_missing_rate_per_omic is applied per selected omic, masking
        randomly chosen currently-observed patient-omic blocks up to the target rate.

        [lifted from a dead nested definition to a real method + fixed by the local
        evaluation_ext copy; see run_blockmissing.py]
        """
        if not (0.0 <= target_missing_rate_per_omic <= 1.0):
            raise ValueError("target_missing_rate_per_omic must be in [0, 1]")
        omics = list(omics) if omics is not None else self.omics

        rng = np.random.default_rng(random_state)
        new_views = {}
        for omic, df in self.views.items():
            new_df = df.copy()
            if omic in omics:
                avail = ~new_df.isna().all(axis=1)
                current_missing = 1.0 - float(avail.mean())
                if target_missing_rate_per_omic > current_missing:
                    # mask additional patients to reach the target missingness
                    need = target_missing_rate_per_omic - current_missing
                    n_mask = int(round(need * len(new_df)))
                    candidate = new_df.index[avail].to_numpy()
                    if len(candidate) > 0 and n_mask > 0:
                        to_mask = rng.choice(candidate, size=min(n_mask, len(candidate)), replace=False)
                        new_df.loc[to_mask, :] = np.nan
            new_views[omic] = new_df

        return MultiOmicsDataset(
            name=self.name,
            views=new_views,
            y=self.y.copy(),
            patient_ids=list(self.patient_ids),
            metadata=dict(self.metadata),
        )


def _make_feature_names(omic: str, n_features: int) -> List[str]:
    return [f"{omic}__f{j:05d}" for j in range(n_features)]


def generate_synthetic_tcga_like_dataset(
    name: Literal["TCGA-BRCA", "TCGA-GBM"],
    n_patients: int,
    omics_dims: Dict[str, int],
    n_classes: int = 3,
    class_probs: Optional[Sequence[float]] = None,
    base_view_missing_rate: float = 0.25,
    missingness_heterogeneity: float = 0.15,
    signal_strength: float = 1.0,
    random_state: int = 0,
) -> MultiOmicsDataset:
    """Generate a synthetic TCGA-like multi-omics dataset.

    Properties of the synthetic data:
    - Multiple omics with different dimensionalities.
    - Multiclass labels with optional class imbalance.
    - Block-wise missingness across views (some patients missing entire modalities).
    - Mild view-specific signal: some features are shifted by class.

    This function is purely for demonstration; real TCGA matrices can be plugged in
    by constructing `MultiOmicsDataset` directly.
    """
    rng = np.random.default_rng(random_state)

    if class_probs is None:
        # Slight imbalance by default
        raw = np.linspace(1.0, 0.6, n_classes)
        class_probs = (raw / raw.sum()).tolist()
    class_probs = np.asarray(class_probs, dtype=float)
    class_probs = class_probs / class_probs.sum()

    # Patient IDs
    patient_ids = [f"{name.split('-')[-1]}_{i:04d}" for i in range(1, n_patients + 1)]

    # Labels (multiclass)
    y = rng.choice(np.arange(n_classes), size=n_patients, p=class_probs)
    y = pd.Series(y, index=pd.Index(patient_ids, name="patient_id"), name="label")

    # Latent class centroids (shared across omics but with omic-specific projections)
    latent_dim = 10
    class_centroids = rng.normal(0, 1, size=(n_classes, latent_dim)) * signal_strength

    views: Dict[str, pd.DataFrame] = {}
    for omic, d in omics_dims.items():
        # Omic-specific projection from latent space -> features
        W = rng.normal(0, 1, size=(latent_dim, d))
        # Patient latent vectors around class centroid
        Z = class_centroids[y.to_numpy()] + rng.normal(0, 1, size=(n_patients, latent_dim))
        X = Z @ W + rng.normal(0, 1, size=(n_patients, d))

        # Add sparse class-discriminative shift to a subset of features
        n_shift = max(5, int(0.02 * d))
        shift_idx = rng.choice(np.arange(d), size=n_shift, replace=False)
        y_arr = y.to_numpy()
        for c in range(n_classes):
            rows = np.where(y_arr == c)[0]
            if len(rows) > 0:
                X[np.ix_(rows, shift_idx)] += (c - (n_classes - 1) / 2.0) * signal_strength

        cols = _make_feature_names(omic, d)
        df = pd.DataFrame(X, index=y.index, columns=cols)

        # Block-wise missingness: some patients have the entire view masked
        # Each view gets a slightly different missing rate
        view_missing_rate = float(np.clip(
            base_view_missing_rate + rng.normal(0, missingness_heterogeneity),
            0.0, 0.95
        ))
        mask_patients = rng.choice(df.index, size=int(round(view_missing_rate * n_patients)), replace=False)
        df.loc[mask_patients, :] = np.nan

        views[omic] = df

    return MultiOmicsDataset(
        name=name,
        views=views,
        y=y,
        patient_ids=patient_ids,
        metadata={
            "synthetic": True,
            "n_patients": n_patients,
            "omics_dims": dict(omics_dims),
            "n_classes": n_classes,
            "class_probs": class_probs.tolist(),
            "base_view_missing_rate": base_view_missing_rate,
            "missingness_heterogeneity": missingness_heterogeneity,
            "signal_strength": signal_strength,
            "random_state": random_state,
        },
    )


# ## Leakage-safe preprocessing and feature selection (pluggable)
# 
# Feature selection is **inside the CV pipeline**, so it is fit only on the training folds.
# 
# The framework supports, at minimum:
# 
# - **variance-based selection** (top-k by variance within each omic, or thresholding),
# - **ANOVA (`SelectKBest(f_classif)`)** within each omic,
# - selecting by **absolute number of features per omic** or by **ratio**.
# 
# Future selectors can be added by implementing another transformer with the same interface.
# 

# In[6]:


@dataclass(frozen=True)
class FeatureSelectionSpec:
    """Configuration for per-omic feature selection."""
    method: Literal["anova", "variance"] = "anova"
    # Choose ONE of the following:
    k_per_omic: Optional[int] = None
    ratio_per_omic: Optional[float] = None  # 0<ratio<=1
    # Per-omic ratio override (README 9.3 one-omic-at-a-time sweep): maps omic ->
    # ratio in (0,1]; omics absent from the dict keep ALL their features (100%).
    # Takes precedence over k_per_omic / ratio_per_omic for the omics it names.
    ratio_by_omic: Optional[Dict[str, float]] = None
    # Optional: remove near-constant features first
    variance_threshold: float = 0.0

    def validate(self) -> None:
        if self.k_per_omic is not None and self.ratio_per_omic is not None:
            raise ValueError("Specify only one of k_per_omic or ratio_per_omic")
        if self.ratio_per_omic is not None and not (0.0 < self.ratio_per_omic <= 1.0):
            raise ValueError("ratio_per_omic must be in (0, 1]")


class PandasTransformer(BaseEstimator, TransformerMixin):
    """Wrap an sklearn transformer but keep pandas DataFrame output (index/columns).

    This helps interpretability and debugging without relying on sklearn's `set_output`.
    """
    def __init__(self, transformer: BaseEstimator):
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "PandasTransformer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PandasTransformer expects a pandas DataFrame as input")
        self.columns_in_ = X.columns.to_list()
        self.transformer_ = clone(self.transformer)
        # Some sklearn transformers ignore y; we pass it when accepted.
        try:
            self.transformer_.fit(X.to_numpy(), y)
        except TypeError:
            self.transformer_.fit(X.to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PandasTransformer expects a pandas DataFrame as input")
        arr = self.transformer_.transform(X.to_numpy())
        cols = self.get_feature_names_out(self.columns_in_)
        return pd.DataFrame(arr, index=X.index, columns=cols)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        feats = list(input_features) if input_features is not None else list(getattr(self, "columns_in_", []))
        if hasattr(self.transformer_, "get_feature_names_out"):
            try:
                return list(self.transformer_.get_feature_names_out(feats))
            except Exception:
                return feats
        return feats


class MultiOmicsFeatureSelector(BaseEstimator, TransformerMixin):
    """Per-omic feature selection transformer (leakage-safe in a CV pipeline).

    Parameters
    ----------
    feature_groups:
        Dict mapping omic name -> list of column names belonging to that omic.
    spec:
        FeatureSelectionSpec controlling method and k/ratio.
    """
    def __init__(self, feature_groups: Dict[str, List[str]], spec: FeatureSelectionSpec):
        self.feature_groups = feature_groups
        self.spec = spec

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "MultiOmicsFeatureSelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MultiOmicsFeatureSelector expects a pandas DataFrame as input")
        self.spec.validate()
        if y is None:
            raise ValueError("y is required for supervised selection (anova)")

        self.selected_feature_names_ = []
        self._per_omic_selected_ = {}

        for omic, cols in self.feature_groups.items():
            cols = [c for c in cols if c in X.columns]
            if not cols:
                continue

            Xo = X[cols].to_numpy()
            # Optional variance threshold
            vt = None
            if self.spec.variance_threshold and self.spec.variance_threshold > 0.0:
                vt = VarianceThreshold(threshold=self.spec.variance_threshold).fit(Xo)
                Xo_v = vt.transform(Xo)
                cols_v = [c for c, keep in zip(cols, vt.get_support()) if keep]
            else:
                Xo_v = Xo
                cols_v = cols

            if len(cols_v) == 0:
                self._per_omic_selected_[omic] = []
                continue

            # Determine k for this omic
            if self.spec.ratio_by_omic is not None and omic in self.spec.ratio_by_omic:
                k = max(1, int(np.floor(float(self.spec.ratio_by_omic[omic]) * len(cols_v))))
            elif self.spec.ratio_by_omic is not None:
                # per-omic sweep: omics not named keep all their features (100%)
                k = len(cols_v)
            elif self.spec.k_per_omic is not None:
                k = int(self.spec.k_per_omic)
            elif self.spec.ratio_per_omic is not None:
                k = max(1, int(np.floor(self.spec.ratio_per_omic * len(cols_v))))
            else:
                # Default: keep all features for this omic
                k = len(cols_v)

            k = min(k, len(cols_v))

            if self.spec.method == "anova":
                skb = SelectKBest(score_func=f_classif, k=k).fit(Xo_v, y)
                keep_mask = skb.get_support()
                selected = [c for c, keep in zip(cols_v, keep_mask) if keep]
                self._per_omic_selector_ = getattr(self, "_per_omic_selector_", {})
                self._per_omic_selector_[omic] = (vt, skb)
            elif self.spec.method == "variance":
                # Select top-k by variance (after thresholding if provided)
                vars_ = np.nanvar(Xo_v, axis=0)
                top_idx = np.argsort(vars_)[::-1][:k]
                selected = [cols_v[i] for i in top_idx]
                self._per_omic_selector_ = getattr(self, "_per_omic_selector_", {})
                self._per_omic_selector_[omic] = (vt, None)
            else:
                raise ValueError(f"Unknown method: {self.spec.method}")

            self._per_omic_selected_[omic] = selected
            self.selected_feature_names_.extend(selected)

        # Keep the original order as they appear in X
        self.selected_feature_names_ = [c for c in X.columns if c in set(self.selected_feature_names_)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MultiOmicsFeatureSelector expects a pandas DataFrame as input")
        return X.loc[:, self.selected_feature_names_].copy()

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> List[str]:
        return list(getattr(self, "selected_feature_names_", []))

    @property
    def per_omic_selected(self) -> Dict[str, List[str]]:
        return dict(getattr(self, "_per_omic_selected_", {}))


# ## Base model abstraction and nested-CV evaluation engine (no leakage)
# 
# `BaseModel` defines a reusable interface and implements shared benchmarking logic:
# 
# - cohort selection and tabular preparation (delegated to `MultiOmicsDataset`),
# - stratified outer CV,
# - leakage-safe inner hyperparameter search (on training folds only),
# - runtime measurement,
# - metric aggregation,
# - structured results.
# 
# Concrete subclasses override:
# - the estimator and pipeline components,
# - the hyperparameter search space,
# - interpretability extraction.
# 

# In[7]:


@dataclass(frozen=True)
class CVSpec:
    outer_splits: int = 5
    inner_splits: int = 3
    random_state: int = 0
    shuffle: bool = True


@dataclass(frozen=True)
class MissingnessSpec:
    missing_policy: Literal["intersection", "impute"] = "impute"
    include_non_intersection_frac: float = 0.0  # used when missing_policy='impute'

@dataclass(frozen=True)
class FusionSpec:
    voting: Literal["hard", "soft"] = "soft"
    score_method: Literal["decision_function", "predict_proba"] = "decision_function"
    weights: Optional[Dict[str, float]] = None  # per-omic weights
    on_missing_view: Literal["skip", "impute"] = "skip"
    tie_break: Literal["score_sum", "lowest_class"] = "score_sum"


@dataclass(frozen=True)
class ExperimentSpec:
    dataset_name: str
    omics: Tuple[str, ...]
    cv: CVSpec
    missingness: MissingnessSpec
    feature_selection: FeatureSelectionSpec
    n_patients: Optional[int] = None  # optional subsampling
    notes: Optional[str] = None

@dataclass
class PreparedExperimentData:
    patient_ids: List[str]
    y: np.ndarray
    n_features_input: int


@dataclass
class TabularPreparedData(PreparedExperimentData):
    X: pd.DataFrame
    feature_groups: Dict[str, List[str]]


@dataclass
class MultiViewPreparedData(PreparedExperimentData):
    X_by_view: Dict[str, pd.DataFrame]
    availability: pd.DataFrame


def _stratified_subsample(
    X: pd.DataFrame,
    y: np.ndarray,
    n: int,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Stratified subsampling without leakage (before CV)."""
    if n >= len(y):
        return X, y
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    indices = np.arange(len(y))
    chosen = []
    for cls in np.unique(y):
        cls_idx = indices[y == cls]
        # keep approximately the same class proportions
        n_cls = max(1, int(round(n * (len(cls_idx) / len(y)))))
        n_cls = min(n_cls, len(cls_idx))
        chosen.extend(rng.choice(cls_idx, size=n_cls, replace=False).tolist())
    chosen = np.array(sorted(set(chosen)))
    if len(chosen) > n:
        chosen = rng.choice(chosen, size=n, replace=False)
    return X.iloc[chosen].copy(), y[chosen].copy()

def _stratified_subsample_patient_ids(
    y: pd.Series,
    n: int,
    random_state: int = 0,
) -> List[str]:
    if n >= len(y):
        return y.index.tolist()

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(y))
    y_arr = y.to_numpy()

    chosen = []
    for cls in np.unique(y_arr):
        cls_idx = indices[y_arr == cls]
        n_cls = max(1, int(round(n * (len(cls_idx) / len(y_arr)))))
        n_cls = min(n_cls, len(cls_idx))
        chosen.extend(rng.choice(cls_idx, size=n_cls, replace=False).tolist())

    chosen = np.array(sorted(set(chosen)))
    if len(chosen) > n:
        chosen = rng.choice(chosen, size=n, replace=False)

    return y.iloc[chosen].index.tolist()

class BaseModel(ABC):
    """Abstract base class for leakage-safe multi-omics benchmarking models."""

    def __init__(
        self,
        cv_spec: CVSpec = CVSpec(),
        imputer: Optional[BaseEstimator] = None,
        scaler: Optional[BaseEstimator] = None,
        feature_selection_spec: FeatureSelectionSpec = FeatureSelectionSpec(),
        store_interpretability_per_fold: bool = False,
        top_n_interpret_features: int = 15,
        n_jobs: int = 1,
        integration: Literal["early", "late"] = "early",
        fusion_spec: Optional[FusionSpec] = None,
    ):
        self.cv_spec = cv_spec
        self.imputer = imputer if imputer is not None else SimpleImputer(strategy="median")
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.feature_selection_spec = feature_selection_spec
        self.store_interpretability_per_fold = store_interpretability_per_fold
        self.top_n_interpret_features = top_n_interpret_features
        self.n_jobs = int(n_jobs)

        if integration not in ("early", "late"):
          raise ValueError("integration must be 'early' or 'late'")

        self.integration = integration
        self.fusion_spec = fusion_spec if fusion_spec is not None else FusionSpec()

    # ---- Subclass hooks ----
    @abstractmethod
    def build_estimator(self) -> BaseEstimator:
        """Return the base estimator (unfitted)."""
        raise NotImplementedError

    @abstractmethod
    def hyperparameter_grid(self) -> List[Dict[str, Any]]:
        """Return a GridSearchCV param_grid (list of dicts)."""
        raise NotImplementedError

    def fixed_hyperparameters(self) -> Dict[str, Any]:
        """
        Optional fast-path for benchmarking:
        subclasses can override this to bypass inner GridSearchCV and fit
        with a fixed estimator configuration inside each outer fold.
        """
        return {}

    def supports_interpretability(self, fitted_pipeline: Pipeline) -> bool:
        return False

    def extract_interpretability(self, fitted_pipeline: Pipeline) -> InterpretabilityResult:
        return InterpretabilityResult(
            supported=False,
            method="none",
            notes="Interpretability not implemented for this model.",
        )

    def prepare_experiment_data(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        patient_ids: Sequence[str],
        missing_policy: Literal["intersection", "impute"] = "impute",
    ) -> PreparedExperimentData:
        if self.integration == "early":
            X, y, groups = dataset.to_tabular(
                omics=omics,
                patient_ids=patient_ids,
                missing_policy=missing_policy,
            )
            return TabularPreparedData(
                patient_ids=list(X.index),
                y=y,
                n_features_input=int(X.shape[1]),
                X=X,
                feature_groups=groups,
            )

        # late integration
        X_by_view, y, availability = dataset.to_multiview(
            omics=omics,
            patient_ids=patient_ids,
            missing_policy=missing_policy,
        )
        return MultiViewPreparedData(
            patient_ids=list(availability.index),
            y=y,
            n_features_input=int(sum(df.shape[1] for df in X_by_view.values())),
            X_by_view=X_by_view,
            availability=availability,
        )

    # ---- Shared utilities ----
    def _build_pipeline(
        self,
        feature_groups: Dict[str, List[str]],
        feature_selection_spec: Optional[FeatureSelectionSpec] = None,
    ) -> Pipeline:
        """Construct a leakage-safe sklearn Pipeline."""
        spec = feature_selection_spec if feature_selection_spec is not None else self.feature_selection_spec
        selector = MultiOmicsFeatureSelector(feature_groups=feature_groups, spec=spec)
        pipe = Pipeline(steps=[
            ("imputer", PandasTransformer(self.imputer)),
            ("selector", selector),
            ("scaler", PandasTransformer(self.scaler)),
            ("estimator", self.build_estimator()),
        ])
        return pipe
    def _fit_single_view_model(
        self,
        X_train_view: pd.DataFrame,
        y_train_view: np.ndarray,
        omic: str,
        experiment_spec: ExperimentSpec,
        inner_cv: StratifiedKFold,
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        feature_groups = {omic: X_train_view.columns.tolist()}
        pipeline = self._build_pipeline(
            feature_groups=feature_groups,
            feature_selection_spec=experiment_spec.feature_selection,
        )

        classes, counts = np.unique(y_train_view, return_counts=True)
        if len(classes) < 2 or counts.min() < inner_cv.n_splits:
            pipeline.fit(X_train_view, y_train_view)
            return pipeline, {"untuned_default_fit": True}

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.hyperparameter_grid(),
            scoring="balanced_accuracy",
            n_jobs=self.n_jobs,
            cv=inner_cv,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train_view, y_train_view)
        return search.best_estimator_, dict(search.best_params_)

    def _get_view_output(
        self,
        fitted_pipeline: Pipeline,
        X_view: pd.DataFrame,
    ) -> Tuple[str, np.ndarray, np.ndarray]:
        est = fitted_pipeline.named_steps["estimator"]
        model_classes = np.asarray(est.classes_)

        if self.fusion_spec.voting == "hard":
            labels = np.asarray(fitted_pipeline.predict(X_view))
            return "hard", labels, model_classes

        # soft voting
        if self.fusion_spec.score_method == "predict_proba":
            probs = np.asarray(fitted_pipeline.predict_proba(X_view))
            return "soft", probs, model_classes

        scores = np.asarray(fitted_pipeline.decision_function(X_view))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return "soft", scores, model_classes

    def _align_scores(
    self,
    scores: np.ndarray,
    model_classes: np.ndarray,
    all_classes: np.ndarray,
    ) -> np.ndarray:
        aligned = np.zeros((scores.shape[0], len(all_classes)), dtype=float)
        for j, cls in enumerate(model_classes):
            col = np.where(all_classes == cls)[0][0]
            aligned[:, col] = scores[:, j]
        return aligned
    def _fuse_predictions(
        self,
        per_view_outputs: Dict[str, Dict[str, Any]],
        all_classes: np.ndarray,
        fallback_class: Any,
        n_test: int,
    ) -> np.ndarray:
        preds = np.full(n_test, fallback_class, dtype=all_classes.dtype)
        weights = self.fusion_spec.weights or {}

        if not per_view_outputs:
            return preds

        if self.fusion_spec.voting == "hard":
            vote_sum = np.zeros((n_test, len(all_classes)), dtype=float)
            contrib = np.zeros(n_test, dtype=float)

            for omic, payload in per_view_outputs.items():
                w = float(weights.get(omic, 1.0))
                idx = np.where(payload["mask"])[0]

                if payload["kind"] == "hard":
                    labels = np.asarray(payload["output"])
                    for row_pos, label in zip(idx, labels):
                        cls_col = np.where(all_classes == label)[0][0]
                        vote_sum[row_pos, cls_col] += w
                    contrib[idx] += w
                else:
                    scores = np.asarray(payload["output"])
                    label_idx = np.argmax(scores, axis=1)
                    for row_pos, cls_col in zip(idx, label_idx):
                        vote_sum[row_pos, cls_col] += w
                    contrib[idx] += w

            valid = contrib > 0
            preds[valid] = all_classes[np.argmax(vote_sum[valid], axis=1)]
            return preds

        # soft voting
        score_sum = np.zeros((n_test, len(all_classes)), dtype=float)
        contrib = np.zeros(n_test, dtype=float)

        for omic, payload in per_view_outputs.items():
            w = float(weights.get(omic, 1.0))
            idx = np.where(payload["mask"])[0]

            if payload["kind"] == "soft":
                vals = np.asarray(payload["output"])
            else:
                labels = np.asarray(payload["output"])
                vals = np.zeros((len(idx), len(all_classes)), dtype=float)
                for r, label in enumerate(labels):
                    cls_col = np.where(all_classes == label)[0][0]
                    vals[r, cls_col] = 1.0

            score_sum[idx] += w * vals
            contrib[idx] += w

        valid = contrib > 0
        preds[valid] = all_classes[np.argmax(score_sum[valid], axis=1)]
        return preds

    def _nested_cv_evaluate_tabular(
        self,
        data: PreparedExperimentData,
        experiment_spec: ExperimentSpec,
        random_state: int = 0,
    ) -> ExperimentResult:
        """Leakage-safe nested CV evaluation with runtime measurement."""
        if not isinstance(data, TabularPreparedData):
            raise NotImplementedError("This model expects tabular prepared data.")

        X = data.X
        y = data.y
        feature_groups = data.feature_groups

        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits,
            shuffle=experiment_spec.cv.shuffle,
            random_state=experiment_spec.cv.random_state,
        )
        inner_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.inner_splits,
            shuffle=experiment_spec.cv.shuffle,
            random_state=experiment_spec.cv.random_state + 1,
        )

        folds: List[FoldResult] = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y[train_idx].copy(), y[test_idx].copy()

            pipeline = self._build_pipeline(feature_groups=feature_groups, feature_selection_spec=experiment_spec.feature_selection)

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.hyperparameter_grid(),
                scoring="balanced_accuracy",
                n_jobs=self.n_jobs,
                cv=inner_cv,
                refit=True,
                error_score="raise",
            )

            t0 = time.perf_counter()
            search.fit(X_train, y_train)
            t_fit = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_pred = search.predict(X_test)
            t_pred = time.perf_counter() - t1

            acc = float(accuracy_score(y_test, y_pred))
            bacc = float(balanced_accuracy_score(y_test, y_pred))

            interpret = None
            if self.store_interpretability_per_fold and self.supports_interpretability(search.best_estimator_):
                try:
                    interpret = self.extract_interpretability(search.best_estimator_)
                except Exception as e:
                    interpret = InterpretabilityResult(
                        supported=False,
                        method="error",
                        notes=f"Interpretability extraction failed: {e}",
                    )

            folds.append(FoldResult(
                fold=fold_idx,
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                accuracy=acc,
                balanced_accuracy=bacc,
                fit_time_sec=float(t_fit),
                predict_time_sec=float(t_pred),
                total_time_sec=float(t_fit + t_pred),
                best_params=dict(search.best_params_),
                interpretability=interpret,
            ))

        metadata = {
            "experiment": asdict(experiment_spec),
            "n_samples": int(len(data.y)),
            "n_features_input": int(data.n_features_input),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "integration": "early"
        }
        return ExperimentResult(folds=folds, metadata=metadata)

    def _nested_cv_evaluate_multiview(
        self,
        data: MultiViewPreparedData,
        experiment_spec: ExperimentSpec,
        random_state: int = 0,
    ) -> ExperimentResult:
        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits,
            shuffle=experiment_spec.cv.shuffle,
            random_state=experiment_spec.cv.random_state,
        )
        inner_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.inner_splits,
            shuffle=experiment_spec.cv.shuffle,
            random_state=experiment_spec.cv.random_state + 1,
        )

        folds: List[FoldResult] = []
        y = data.y
        all_classes_global = np.unique(y)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
            y_train, y_test = y[train_idx].copy(), y[test_idx].copy()
            fallback_class = pd.Series(y_train).value_counts().idxmax()

            fitted_views: Dict[str, Pipeline] = {}
            best_params: Dict[str, Any] = {}

            t0 = time.perf_counter()

            for omic, X_view_full in data.X_by_view.items():
                X_train_view = X_view_full.iloc[train_idx].copy()
                train_avail = data.availability[omic].iloc[train_idx]

                if self.fusion_spec.on_missing_view == "skip":
                    if int(train_avail.sum()) == 0:
                        best_params[omic] = {"skipped": "no_available_train_samples"}
                        continue
                    X_train_fit = X_train_view.loc[train_avail].copy()
                    y_train_fit = y_train[train_avail.to_numpy()].copy()
                else:
                    if int(train_avail.sum()) == 0:
                        best_params[omic] = {"skipped": "no_available_train_samples"}
                        continue
                    X_train_fit = X_train_view.copy()
                    y_train_fit = y_train.copy()

                if np.unique(y_train_fit).size < 2:
                    best_params[omic] = {"skipped": "fewer_than_2_classes"}
                    continue

                try:
                    fitted_pipe, params = self._fit_single_view_model(
                        X_train_view=X_train_fit,
                        y_train_view=y_train_fit,
                        omic=omic,
                        experiment_spec=experiment_spec,
                        inner_cv=inner_cv,
                    )
                    fitted_views[omic] = fitted_pipe
                    best_params[omic] = params
                except Exception as e:
                    best_params[omic] = {"skipped": f"fit_failed: {e}"}

            t_fit = time.perf_counter() - t0

            t1 = time.perf_counter()

            per_view_outputs: Dict[str, Dict[str, Any]] = {}
            n_test = len(test_idx)

            for omic, fitted_pipe in fitted_views.items():
                X_test_view = data.X_by_view[omic].iloc[test_idx].copy()
                test_avail = data.availability[omic].iloc[test_idx]

                if self.fusion_spec.on_missing_view == "skip":
                    mask = test_avail.to_numpy()
                    if int(mask.sum()) == 0:
                        continue
                    X_test_eval = X_test_view.loc[test_avail].copy()
                else:
                    mask = np.ones(n_test, dtype=bool)
                    X_test_eval = X_test_view.copy()

                try:
                    kind, output, model_classes = self._get_view_output(fitted_pipe, X_test_eval)

                    if kind == "soft":
                        output = self._align_scores(
                            scores=np.asarray(output),
                            model_classes=np.asarray(model_classes),
                            all_classes=all_classes_global,
                        )

                    per_view_outputs[omic] = {
                        "mask": mask,
                        "kind": kind,
                        "output": output,
                    }
                except Exception:
                    continue

            y_pred = self._fuse_predictions(
                per_view_outputs=per_view_outputs,
                all_classes=all_classes_global,
                fallback_class=fallback_class,
                n_test=n_test,
            )

            t_pred = time.perf_counter() - t1

            acc = float(accuracy_score(y_test, y_pred))
            bacc = float(balanced_accuracy_score(y_test, y_pred))

            best_params["fusion"] = asdict(self.fusion_spec)

            folds.append(FoldResult(
                fold=fold_idx,
                n_train=int(len(train_idx)),
                n_test=int(len(test_idx)),
                accuracy=acc,
                balanced_accuracy=bacc,
                fit_time_sec=float(t_fit),
                predict_time_sec=float(t_pred),
                total_time_sec=float(t_fit + t_pred),
                best_params=best_params,
                interpretability=None,
            ))

        metadata = {
            "experiment": asdict(experiment_spec),
            "n_samples": int(len(data.y)),
            "n_features_input": int(data.n_features_input),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "integration": "late",
        }
        return ExperimentResult(folds=folds, metadata=metadata)

    def _nested_cv_evaluate(
        self,
        data: PreparedExperimentData,
        experiment_spec: ExperimentSpec,
        random_state: int = 0,
    ) -> ExperimentResult:
        if isinstance(data, TabularPreparedData):
            return self._nested_cv_evaluate_tabular(
                data=data,
                experiment_spec=experiment_spec,
                random_state=random_state,
            )

        if isinstance(data, MultiViewPreparedData):
            return self._nested_cv_evaluate_multiview(
                data=data,
                experiment_spec=experiment_spec,
                random_state=random_state,
            )

        raise TypeError(f"Unsupported prepared data type: {type(data).__name__}")

    # ---- Public benchmarking API (framework-level) ----
    def evaluate_on_missing_patients(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        include_non_intersection_fracs: Sequence[float],
        feature_selection_spec: Optional[FeatureSelectionSpec] = None,
        missing_policy: Literal["intersection", "impute"] = "impute",
        n_patients: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> GridResult:
        """Benchmark **accuracy vs missingness** by varying the fraction of incomplete patients included."""
        axis = list(include_non_intersection_fracs)
        results: Dict[float, ExperimentResult] = {}
        for frac in axis:
            cohort = dataset.build_cohort(omics=omics, include_non_intersection_frac=frac, random_state=self.cv_spec.random_state)

            if n_patients is not None:
                cohort = _stratified_subsample_patient_ids(
                    dataset.y.loc[cohort],
                    n=n_patients,
                    random_state=self.cv_spec.random_state,
                )

            prepared = self.prepare_experiment_data(
                dataset=dataset,
                omics=omics,
                patient_ids=cohort,
                missing_policy=missing_policy,
            )

            exp_spec = ExperimentSpec(
                dataset_name=dataset.name,
                omics=tuple(omics),
                cv=self.cv_spec,
                missingness=MissingnessSpec(missing_policy=missing_policy, include_non_intersection_frac=float(frac)),
                feature_selection=feature_selection_spec or self.feature_selection_spec,
                n_patients=int(n_patients) if n_patients is not None else None,
                notes=notes,
            )
            results[float(frac)] = self._nested_cv_evaluate(
                prepared,
                exp_spec,
                random_state=self.cv_spec.random_state,
            )

        return GridResult(results=results, axis_name="include_non_intersection_frac", axis_values=axis)

    def evaluate_on_n_features(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        k_values: Sequence[int],
        selection_method: Literal["anova", "variance"] = "anova",
        missing_policy: Literal["intersection", "impute"] = "impute",
        include_non_intersection_frac: float = 0.0,
        n_patients: Optional[int] = None,
        variance_threshold: float = 0.0,
        notes: Optional[str] = None,
    ) -> GridResult:
        """Benchmark **accuracy vs number of features** (per omic)."""
        axis = list(k_values)
        results: Dict[int, ExperimentResult] = {}

        cohort = dataset.build_cohort(
            omics=omics,
            include_non_intersection_frac=include_non_intersection_frac,
            random_state=self.cv_spec.random_state,
        )

        if n_patients is not None:
            cohort = _stratified_subsample_patient_ids(
                dataset.y.loc[cohort],
                n=n_patients,
                random_state=self.cv_spec.random_state,
            )

        for k in axis:
            fs = FeatureSelectionSpec(
                method=selection_method,
                k_per_omic=int(k),
                ratio_per_omic=None,
                variance_threshold=variance_threshold,
            )

            exp_spec = ExperimentSpec(
                dataset_name=dataset.name,
                omics=tuple(omics),
                cv=self.cv_spec,
                missingness=MissingnessSpec(
                    missing_policy=missing_policy,
                    include_non_intersection_frac=float(include_non_intersection_frac),
                ),
                feature_selection=fs,
                n_patients=int(n_patients) if n_patients is not None else None,
                notes=notes,
            )

            prepared = self.prepare_experiment_data(
                dataset=dataset,
                omics=omics,
                patient_ids=cohort,
                missing_policy=missing_policy,
            )

            results[int(k)] = self._nested_cv_evaluate(
                prepared,
                exp_spec,
                random_state=self.cv_spec.random_state,
            )

        return GridResult(results=results, axis_name="k_per_omic", axis_values=axis)


    def evaluate_on_feature_ratio(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        ratios: Sequence[float],
        selection_method: Literal["anova", "variance"] = "anova",
        missing_policy: Literal["intersection", "impute"] = "impute",
        include_non_intersection_frac: float = 0.0,
        n_patients: Optional[int] = None,
        variance_threshold: float = 0.0,
        notes: Optional[str] = None,
    ) -> GridResult:
        """Benchmark **accuracy vs ratio of selected features** (per omic)."""
        axis = [float(r) for r in ratios]
        results: Dict[float, ExperimentResult] = {}

        cohort = dataset.build_cohort(
            omics=omics,
            include_non_intersection_frac=include_non_intersection_frac,
            random_state=self.cv_spec.random_state,
        )

        if n_patients is not None:
            cohort = _stratified_subsample_patient_ids(
                dataset.y.loc[cohort],
                n=n_patients,
                random_state=self.cv_spec.random_state,
            )

        for r in axis:
            fs = FeatureSelectionSpec(
                method=selection_method,
                k_per_omic=None,
                ratio_per_omic=float(r),
                variance_threshold=variance_threshold,
            )

            exp_spec = ExperimentSpec(
                dataset_name=dataset.name,
                omics=tuple(omics),
                cv=self.cv_spec,
                missingness=MissingnessSpec(
                    missing_policy=missing_policy,
                    include_non_intersection_frac=float(include_non_intersection_frac),
                ),
                feature_selection=fs,
                n_patients=int(n_patients) if n_patients is not None else None,
                notes=notes,
            )

            prepared = self.prepare_experiment_data(
                dataset=dataset,
                omics=omics,
                patient_ids=cohort,
                missing_policy=missing_policy,
            )

            results[float(r)] = self._nested_cv_evaluate(
                prepared,
                exp_spec,
                random_state=self.cv_spec.random_state,
            )

        return GridResult(results=results, axis_name="ratio_per_omic", axis_values=axis)


    def evaluate_on_feature_ratio_one_omic(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        target_omic: str,
        ratios: Sequence[float],
        selection_method: Literal["anova", "variance"] = "anova",
        missing_policy: Literal["intersection", "impute"] = "impute",
        include_non_intersection_frac: float = 0.0,
        n_patients: Optional[int] = None,
        variance_threshold: float = 0.0,
        notes: Optional[str] = None,
    ) -> GridResult:
        """README 9.3 one-omic-at-a-time sweep: vary the retained feature ratio of
        `target_omic` while keeping every other omic at 100% of its features."""
        if target_omic not in set(omics):
            raise ValueError(f"target_omic {target_omic!r} not in omics {list(omics)}")
        axis = [float(r) for r in ratios]
        results: Dict[float, ExperimentResult] = {}

        cohort = dataset.build_cohort(
            omics=omics,
            include_non_intersection_frac=include_non_intersection_frac,
            random_state=self.cv_spec.random_state,
        )
        if n_patients is not None:
            cohort = _stratified_subsample_patient_ids(
                dataset.y.loc[cohort], n=n_patients, random_state=self.cv_spec.random_state,
            )

        for r in axis:
            fs = FeatureSelectionSpec(
                method=selection_method,
                k_per_omic=None,
                ratio_per_omic=None,
                ratio_by_omic={target_omic: float(r)},
                variance_threshold=variance_threshold,
            )
            exp_spec = ExperimentSpec(
                dataset_name=dataset.name,
                omics=tuple(omics),
                cv=self.cv_spec,
                missingness=MissingnessSpec(
                    missing_policy=missing_policy,
                    include_non_intersection_frac=float(include_non_intersection_frac),
                ),
                feature_selection=fs,
                n_patients=int(n_patients) if n_patients is not None else None,
                notes=notes,
            )
            prepared = self.prepare_experiment_data(
                dataset=dataset, omics=omics, patient_ids=cohort, missing_policy=missing_policy,
            )
            results[float(r)] = self._nested_cv_evaluate(
                prepared, exp_spec, random_state=self.cv_spec.random_state,
            )

        return GridResult(results=results, axis_name=f"ratio_{target_omic}", axis_values=axis)


    def evaluate_on_n_patients(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        n_values: Sequence[int],
        missing_policy: Literal["intersection", "impute"] = "impute",
        include_non_intersection_frac: float = 0.0,
        feature_selection_spec: Optional[FeatureSelectionSpec] = None,
        notes: Optional[str] = None,
    ) -> GridResult:
        """Benchmark **accuracy vs number of patients** (stratified subsampling)."""
        axis = [int(n) for n in n_values]
        results: Dict[int, ExperimentResult] = {}

        base_cohort = dataset.build_cohort(
            omics=omics,
            include_non_intersection_frac=include_non_intersection_frac,
            random_state=self.cv_spec.random_state,
        )

        for n in axis:
            cohort = _stratified_subsample_patient_ids(
                dataset.y.loc[base_cohort],
                n=n,
                random_state=self.cv_spec.random_state,
            )

            exp_spec = ExperimentSpec(
                dataset_name=dataset.name,
                omics=tuple(omics),
                cv=self.cv_spec,
                missingness=MissingnessSpec(
                    missing_policy=missing_policy,
                    include_non_intersection_frac=float(include_non_intersection_frac),
                ),
                feature_selection=feature_selection_spec or self.feature_selection_spec,
                n_patients=int(n),
                notes=notes,
            )

            prepared = self.prepare_experiment_data(
                dataset=dataset,
                omics=omics,
                patient_ids=cohort,
                missing_policy=missing_policy,
            )

            results[int(n)] = self._nested_cv_evaluate(
                prepared,
                exp_spec,
                random_state=self.cv_spec.random_state,
            )

        return GridResult(results=results, axis_name="n_patients", axis_values=axis)

    def evaluate_on_n_modalities(
        self,
        dataset: MultiOmicsDataset,
        modality_sets: Sequence[Sequence[str]],
        missing_policy: Literal["intersection", "impute"] = "impute",
        include_non_intersection_frac: float = 0.0,
        n_patients: Optional[int] = None,
        feature_selection_spec: Optional[FeatureSelectionSpec] = None,
        notes: Optional[str] = None,
    ) -> GridResult:
        """Benchmark **accuracy vs number of modalities** by evaluating specific modality subsets."""
        axis = [tuple(ms) for ms in modality_sets]
        results: Dict[Tuple[str, ...], ExperimentResult] = {}

        for ms in axis:
            cohort = dataset.build_cohort(omics=ms, include_non_intersection_frac=include_non_intersection_frac, random_state=self.cv_spec.random_state)
            if n_patients is not None:
                cohort = _stratified_subsample_patient_ids(
                    dataset.y.loc[cohort],
                    n=n_patients,
                    random_state=self.cv_spec.random_state,
                )

            prepared = self.prepare_experiment_data(
                dataset=dataset,
                omics=ms,
                patient_ids=cohort,
                missing_policy=missing_policy,
)

            exp_spec = ExperimentSpec(
                dataset_name=dataset.name,
                omics=tuple(ms),
                cv=self.cv_spec,
                missingness=MissingnessSpec(missing_policy=missing_policy, include_non_intersection_frac=float(include_non_intersection_frac)),
                feature_selection=feature_selection_spec or self.feature_selection_spec,
                n_patients=int(n_patients) if n_patients is not None else None,
                notes=notes,
            )
            results[tuple(ms)] = self._nested_cv_evaluate(
                    prepared,
                    exp_spec,
                    random_state=self.cv_spec.random_state,
                )

        return GridResult(results=results, axis_name="modalities", axis_values=axis)

    def evaluate_on_missingness_ratio(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        target_missing_rates: Sequence[float],
        missing_policy: Literal["intersection", "impute"] = "impute",
        include_non_intersection_frac: float = 1.0,
        n_patients: Optional[int] = None,
        feature_selection_spec: Optional[FeatureSelectionSpec] = None,
        notes: Optional[str] = None,
    ) -> GridResult:
        """Benchmark **accuracy vs missingness ratio** by *simulating* additional missingness (demo helper)."""
        axis = [float(r) for r in target_missing_rates]
        results: Dict[float, ExperimentResult] = {}

        for r in axis:
            ds_r = dataset.with_additional_view_missingness(target_missing_rate_per_omic=float(r), omics=omics, random_state=self.cv_spec.random_state)
            cohort = ds_r.build_cohort(omics=omics, include_non_intersection_frac=include_non_intersection_frac, random_state=self.cv_spec.random_state)
            if n_patients is not None:
                cohort = _stratified_subsample_patient_ids(
                    dataset.y.loc[cohort],
                    n=n_patients,
                    random_state=self.cv_spec.random_state,
                )

            prepared = self.prepare_experiment_data(
                dataset=ds_r,
                omics=omics,
                patient_ids=cohort,
                missing_policy=missing_policy,
            )

            exp_spec = ExperimentSpec(
                dataset_name=ds_r.name,
                omics=tuple(omics),
                cv=self.cv_spec,
                missingness=MissingnessSpec(missing_policy=missing_policy, include_non_intersection_frac=float(include_non_intersection_frac)),
                feature_selection=feature_selection_spec or self.feature_selection_spec,
                n_patients=int(n_patients) if n_patients is not None else None,
                notes=notes,
            )
            results[float(r)] = self._nested_cv_evaluate(
                    prepared,
                    exp_spec,
                    random_state=self.cv_spec.random_state,
                )

        return GridResult(results=results, axis_name="target_missing_rate_per_omic", axis_values=axis)

    def benchmark_runtime(
        self,
        grid_result: GridResult,
        metric: Literal["total_time_mean_sec", "total_time_std_sec"] = "total_time_mean_sec",
    ) -> pd.DataFrame:
        """Convenience helper for the **accuracy vs computation time** axis."""
        df = grid_result.summary_frame()
        # Keep only time + primary accuracy metric for quick inspection
        cols = [grid_result.axis_name, "balanced_accuracy_mean", "balanced_accuracy_std", metric]
        return df[cols].copy()






