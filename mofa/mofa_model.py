from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation"))
from base import (
    BaseModel,
    CVSpec,
    ExperimentResult,
    ExperimentSpec,
    FeatureSelectionSpec,
    FoldResult,
    MultiOmicsDataset,
    TabularPreparedData,
)


def load_compat_dataset(path: str) -> MultiOmicsDataset:
    import joblib

    p = joblib.load(path)
    views = {
        omic: pd.DataFrame(
            d["values"],
            index=pd.Index(d["index"], dtype=object),
            columns=pd.Index(d["columns"], dtype=object),
        )
        for omic, d in p["views"].items()
    }
    y_idx = pd.Index(p["y"]["index"], dtype=object)
    y = pd.Series(p["y"]["values"], index=y_idx, dtype=object)
    views = {k: v.reindex(y_idx) for k, v in views.items()}
    return MultiOmicsDataset(
        name=p["name"],
        views=views,
        y=y,
        patient_ids=[str(x) for x in p["patient_ids"]],
        metadata=p.get("metadata", {}),
    )


def _anova_select(X_obs: pd.DataFrame, y_obs: np.ndarray, k: int) -> List[str]:
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


class MOFAEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_groups: Dict[str, List[str]],
        n_factors: int = 15,
        use_obs: Literal["union", "intersection"] = "union",
        convergence_mode: str = "medium",
        gpu_mode: bool = False,
        center_views: bool = True,
        scale_views: bool = False,
        projection_reg: float = 1e-6,
        random_state: int = 0,
        temp_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.feature_groups = feature_groups
        self.n_factors = int(n_factors)
        self.use_obs = use_obs
        self.convergence_mode = convergence_mode
        self.gpu_mode = bool(gpu_mode)
        self.center_views = bool(center_views)
        self.scale_views = bool(scale_views)
        self.projection_reg = float(projection_reg)
        self.random_state = int(random_state)
        self.temp_dir = temp_dir
        self.verbose = bool(verbose)

    @staticmethod
    def _lazy_imports():
        import scanpy as sc
        import muon as mu
        import mofax as mfx
        return sc, mu, mfx

    def _split_views(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        by_view = {}
        for omic, cols in self.feature_groups.items():
            cols_here = [c for c in cols if c in X.columns]
            if cols_here:
                by_view[omic] = X[cols_here].copy()
        return by_view

    def _fit_standardization(self, X_by_view: Dict[str, pd.DataFrame]) -> None:
        self.view_means_ = {}
        self.view_stds_ = {}
        for omic, df in X_by_view.items():
            means = df.mean(axis=0, skipna=True)
            stds = df.std(axis=0, skipna=True, ddof=0).replace(0.0, 1.0)
            self.view_means_[omic] = means
            self.view_stds_[omic] = stds

    def _apply_standardization(self, X_by_view: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        out = {}
        for omic, df in X_by_view.items():
            arr = df.copy()
            if self.center_views:
                arr = arr - self.view_means_[omic]
            if self.scale_views:
                arr = arr / self.view_stds_[omic]
            out[omic] = arr.fillna(0.0)
        return out

    def _build_mdata(self, X_by_view: Dict[str, pd.DataFrame]):
        sc, mu, _ = self._lazy_imports()
        mods = {}
        for omic, df in X_by_view.items():
            ad = sc.AnnData(df.to_numpy(dtype=float))
            ad.obs_names = [str(i) for i in df.index]
            ad.var_names = [str(c) for c in df.columns]
            mods[omic] = ad
        return mu.MuData(mods)

    def _train_mofa(self, mdata, outfile: str):
        _, mu, _ = self._lazy_imports()
        kwargs = dict(
            use_obs=self.use_obs,
            n_factors=self.n_factors,
            convergence_mode=self.convergence_mode,
            outfile=outfile,
            gpu_mode=self.gpu_mode,
            seed=self.random_state,
        )
        for keys_to_drop in [(), ("seed",), ("gpu_mode", "seed"), ("convergence_mode", "gpu_mode", "seed")]:
            call_kwargs = {k: v for k, v in kwargs.items() if k not in set(keys_to_drop)}
            try:
                mu.tl.mofa(mdata, **call_kwargs)
                return
            except TypeError:
                continue
        mu.tl.mofa(mdata, use_obs=self.use_obs, n_factors=self.n_factors)

    @staticmethod
    def _factor_columns_from_df(df: pd.DataFrame) -> List[str]:
        cols = [c for c in df.columns if str(c).lower().startswith("factor")]
        return cols if cols else df.columns.tolist()

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "MOFAEmbeddingTransformer":
        _, _, mfx = self._lazy_imports()

        self.input_features_ = X.columns.tolist()
        X_by_view = self._split_views(X)
        self.view_feature_names_ = {omic: df.columns.tolist() for omic, df in X_by_view.items()}

        self._fit_standardization(X_by_view)
        X_proc = self._apply_standardization(X_by_view)

        mdata = self._build_mdata(X_proc)
        self._tmpdir = tempfile.TemporaryDirectory(dir=self.temp_dir)
        self.outfile_ = os.path.join(self._tmpdir.name, "mofa_model.hdf5")

        self._train_mofa(mdata, outfile=self.outfile_)

        if "X_mofa" in mdata.obsm:
            Z_train = np.asarray(mdata.obsm["X_mofa"])
            n_latent = Z_train.shape[1]
            factor_cols = [f"Factor{i+1}" for i in range(n_latent)]
            self.train_factors_ = pd.DataFrame(Z_train, index=X.index, columns=factor_cols)
        else:
            model = mfx.mofa_model(self.outfile_)
            df_f = model.get_factors(df=True)
            factor_cols = self._factor_columns_from_df(df_f)
            self.train_factors_ = df_f[factor_cols].copy()
            self.train_factors_ = self.train_factors_.loc[X.index]
            try:
                model.close()
            except Exception:
                pass

        self.output_features_ = self.train_factors_.columns.tolist()
        self.n_factors_learned_ = len(self.output_features_)
        self.train_index_ = X.index.copy()

        self.weights_ = {}
        weights_from_mdata = True
        for omic, cols in self.view_feature_names_.items():
            try:
                W = np.asarray(mdata.mod[omic].varm["LFs"], dtype=float)
                if W.shape[0] != len(cols):
                    raise ValueError("Unexpected loading matrix shape")
                self.weights_[omic] = W[:, : self.n_factors_learned_]
            except Exception:
                weights_from_mdata = False
                break

        if not weights_from_mdata:
            model = mfx.mofa_model(self.outfile_)
            for omic, cols in self.view_feature_names_.items():
                w_df = None
                for view_key in [omic, omic.lower()]:
                    try:
                        w_df = model.get_weights(views=view_key, df=True)
                        break
                    except Exception:
                        continue
                if w_df is None:
                    raise RuntimeError(f"Could not extract MOFA weights for view '{omic}'.")
                factor_cols = self._factor_columns_from_df(w_df)
                w_df = w_df[factor_cols]
                if list(w_df.index) != cols:
                    common = [c for c in cols if c in w_df.index]
                    if len(common) != len(cols):
                        raise RuntimeError(f"Could not align MOFA weights for view '{omic}'.")
                    w_df = w_df.loc[cols]
                self.weights_[omic] = w_df.to_numpy(dtype=float)[:, : self.n_factors_learned_]
            try:
                model.close()
            except Exception:
                pass

        return self

    def _project_one_sample(self, row: pd.Series) -> np.ndarray:
        AtA = np.zeros((self.n_factors_learned_, self.n_factors_learned_), dtype=float)
        Atx = np.zeros(self.n_factors_learned_, dtype=float)
        has_data = False

        for omic, cols in self.view_feature_names_.items():
            x = row[cols].to_numpy(dtype=float)
            if self.center_views:
                x = x - self.view_means_[omic].reindex(cols).to_numpy(dtype=float)
            if self.scale_views:
                x = x / self.view_stds_[omic].reindex(cols).to_numpy(dtype=float)

            mask = np.isfinite(x)
            if not np.any(mask):
                continue

            W = self.weights_[omic][mask, :]
            xv = x[mask]
            AtA += W.T @ W
            Atx += W.T @ xv
            has_data = True

        if not has_data:
            return np.zeros(self.n_factors_learned_, dtype=float)

        AtA += self.projection_reg * np.eye(self.n_factors_learned_)
        return np.linalg.solve(AtA, Atx)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if X.index.isin(self.train_index_).all():
            missing = [idx for idx in X.index if idx not in self.train_factors_.index]
            if not missing:
                return self.train_factors_.loc[X.index].copy()

        Z = np.vstack([self._project_one_sample(X.loc[idx]) for idx in X.index])
        return pd.DataFrame(Z, index=X.index, columns=self.output_features_)


class MOFAModel(BaseModel):
    def __init__(
        self,
        cv_spec: CVSpec = CVSpec(),
        feature_selection_spec: FeatureSelectionSpec = FeatureSelectionSpec(
            method="anova", k_per_omic=100
        ),
        n_latent: int = 15,
        downstream_c: float = 1.0,
        n_jobs: int = 1,
    ):
        super().__init__(
            cv_spec=cv_spec,
            feature_selection_spec=feature_selection_spec,
            n_jobs=n_jobs,
            integration="early",
        )
        self.n_latent = int(n_latent)
        self.downstream_c = float(downstream_c)

    def build_estimator(self):
        raise NotImplementedError("MOFAModel uses its own training loop.")

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
            omics=omics,
            patient_ids=patient_ids,
            missing_policy=missing_policy,
        )
        return TabularPreparedData(
            patient_ids=list(X.index),
            y=y,
            n_features_input=int(X.shape[1]),
            X=X,
            feature_groups=feature_groups,
        )

    def _run_single_fold(
        self,
        X_train_df: pd.DataFrame,
        y_train: np.ndarray,
        X_test_df: pd.DataFrame,
        y_test: np.ndarray,
        fs_spec: FeatureSelectionSpec,
        feature_groups: Dict[str, List[str]],
        random_state: int,
    ) -> Tuple[np.ndarray, float, float]:
        if fs_spec.k_per_omic is not None:
            use_ratio = False
            k_base = int(fs_spec.k_per_omic)
        elif fs_spec.ratio_per_omic is not None:
            use_ratio = True
            ratio = float(fs_spec.ratio_per_omic)
        else:
            use_ratio = False
            k_base = 200

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

        selected_cols = [c for cols in selected_groups.values() for c in cols]
        if not selected_cols:
            selected_groups = {k: list(v) for k, v in feature_groups.items()}
            selected_cols = [c for cols in selected_groups.values() for c in cols]

        X_train_sel = X_train_df[selected_cols]
        X_test_sel = X_test_df[selected_cols]

        n_factors = min(self.n_latent, max(1, len(selected_cols) - 1), max(1, len(X_train_sel) - 1))

        t0 = time.perf_counter()
        embed = MOFAEmbeddingTransformer(
            feature_groups=selected_groups,
            n_factors=n_factors,
            random_state=random_state,
        )
        embed.fit(X_train_sel, y_train)
        Z_train = embed.transform(X_train_sel)
        Z_test = embed.transform(X_test_sel)

        scaler = StandardScaler()
        Z_train_s = scaler.fit_transform(Z_train.to_numpy(dtype=float))
        Z_test_s = scaler.transform(Z_test.to_numpy(dtype=float))

        clf = LogisticRegression(
            C=self.downstream_c,
            max_iter=5000,
            class_weight="balanced",
        )
        clf.fit(Z_train_s, y_train)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = clf.predict(Z_test_s)
        t_pred = time.perf_counter() - t1

        # Drop the temp hdf5 handle so the fitted embedder pickles cleanly;
        # out-of-sample projection only needs the learned weights/stats.
        embed._tmpdir = None
        embed.outfile_ = None
        artifact = {
            "model_type": "mofa",
            "mofa_embedder": embed,
            "scaler": scaler,
            "classifier": clf,
            "selected_cols": list(selected_cols),
            "selected_groups": {k: list(v) for k, v in selected_groups.items()},
            "n_factors": int(n_factors),
        }

        return y_pred, t_fit, t_pred, artifact

    def _nested_cv_evaluate(
        self,
        data,
        experiment_spec: ExperimentSpec,
        random_state: int = 0,
    ) -> ExperimentResult:
        if not isinstance(data, TabularPreparedData):
            raise TypeError("MOFAModel requires TabularPreparedData.")

        X = data.X
        y = data.y
        feature_groups = data.feature_groups
        fs_spec = experiment_spec.feature_selection
        seed = experiment_spec.cv.random_state

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits,
            shuffle=True,
            random_state=seed,
        )

        folds: List[FoldResult] = []
        self.fold_artifacts_ = []
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_enc), start=1):
            X_train_df = X.iloc[train_idx].copy()
            X_test_df = X.iloc[test_idx].copy()
            y_train = y_enc[train_idx]
            y_test = y_enc[test_idx]

            y_pred, t_fit, t_pred, artifact = self._run_single_fold(
                X_train_df=X_train_df,
                y_train=y_train,
                X_test_df=X_test_df,
                y_test=y_test,
                fs_spec=fs_spec,
                feature_groups=feature_groups,
                random_state=seed,
            )
            artifact["fold"] = int(fold_idx)
            artifact["label_classes"] = le.classes_.tolist()
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
                        "n_latent": self.n_latent,
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
                "integration": "mofa",
            },
        )
