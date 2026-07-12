from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
    InterpretabilityResult,
    MissingnessSpec,
    MultiOmicsDataset,
    MultiViewPreparedData,
    PreparedExperimentData,
)

try:
    from integrao.integrater import integrao_integrater, integrao_predictor
    import torch

    _INTEGRAO_AVAILABLE = True
except ImportError:
    _INTEGRAO_AVAILABLE = False
    warnings.warn(
        "IntegrAO not found. Install with: pip install integrao  "
        "(and PyTorch + torch-geometric).",
        stacklevel=2,
    )


# ── Feature selection helper ──────────────────────────────────────────────────

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


def _anova_select(
    X_obs: pd.DataFrame,
    y_obs: np.ndarray,
    k: int,
) -> List[str]:
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


# ── Model ─────────────────────────────────────────────────────────────────────

class IntegrAOModel(BaseModel):
    """
    IntegrAO (GNN-based patient graph fusion) wrapped in the shared benchmarking interface.

    Supports two modes:
      - 'supervised':   end-to-end GNN classification fine-tuning.
      - 'unsupervised': GNN embedding (unsupervised) + logistic regression on top.

    HPs are fixed at construction time and swept externally via SLURM array jobs
    (same pattern as SVM / LogReg / MOFA in this benchmark).  No inner CV loop
    is used because GNN training is too expensive for nested search.

    Reference: Ma et al., Nature Machine Intelligence, 2025.
               https://github.com/bowang-lab/IntegrAO
    """

    def __init__(
        self,
        cv_spec: CVSpec = CVSpec(),
        feature_selection_spec: FeatureSelectionSpec = FeatureSelectionSpec(
            method="anova", k_per_omic=100
        ),
        mode: Literal["supervised", "unsupervised"] = "supervised",
        neighbor_size: int = 20,
        embedding_dims: int = 64,
        hidden_channels: int = 128,
        fusing_iteration: int = 20,
        align_epochs: int = 1000,
        finetune_epochs: int = 1000,
        mu: float = 0.5,
        normalization_factor: float = 1.0,
        lr_C: float = 1.0,
        n_jobs: int = 1,
    ):
        super().__init__(
            cv_spec=cv_spec,
            feature_selection_spec=feature_selection_spec,
            n_jobs=n_jobs,
            integration="late",
        )
        self.mode = mode
        self.neighbor_size = neighbor_size
        self.embedding_dims = embedding_dims
        self.hidden_channels = hidden_channels
        self.fusing_iteration = fusing_iteration
        self.align_epochs = align_epochs
        self.finetune_epochs = finetune_epochs
        self.mu = mu
        self.normalization_factor = normalization_factor
        self.lr_C = lr_C

    # ── BaseModel abstract stubs (IntegrAO bypasses sklearn pipeline) ─────────

    def build_estimator(self):
        raise NotImplementedError("IntegrAOModel uses its own GNN training loop.")

    def hyperparameter_grid(self):
        return [{}]

    # ── Data preparation ──────────────────────────────────────────────────────

    def prepare_experiment_data(
        self,
        dataset: MultiOmicsDataset,
        omics: Sequence[str],
        patient_ids: Sequence[str],
        missing_policy: Literal["intersection", "impute"] = "impute",
    ) -> MultiViewPreparedData:
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

    # ── Core CV loop ──────────────────────────────────────────────────────────

    def _nested_cv_evaluate(
        self,
        data: PreparedExperimentData,
        experiment_spec: ExperimentSpec,
        random_state: int = 0,
    ) -> ExperimentResult:
        if not isinstance(data, MultiViewPreparedData):
            raise TypeError("IntegrAOModel requires MultiViewPreparedData.")
        if not _INTEGRAO_AVAILABLE:
            raise ImportError(
                "IntegrAO not installed. Run: pip install integrao"
            )

        X_by_view = data.X_by_view
        y_raw = data.y
        omics = sorted(X_by_view.keys())
        fs_spec = experiment_spec.feature_selection

        # 0-index labels for PyTorch CrossEntropyLoss
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        n_classes = int(len(le.classes_))

        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits,
            shuffle=True,
            random_state=experiment_spec.cv.random_state,
        )

        all_ids = list(X_by_view[omics[0]].index)
        folds: List[FoldResult] = []
        self.fold_artifacts_ = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(np.zeros(len(y)), y), start=1
        ):
            t_start = time.perf_counter()

            train_ids = [all_ids[i] for i in train_idx]
            test_ids = [all_ids[i] for i in test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # ── Feature count for this fold ──────────────────────────────────
            if fs_spec.k_per_omic is not None:
                k = int(fs_spec.k_per_omic)
                use_ratio = False
            elif fs_spec.ratio_per_omic is not None:
                use_ratio = True
                ratio = float(fs_spec.ratio_per_omic)
            else:
                k = 200
                use_ratio = False

            # ── Per-omic ANOVA feature selection on train ────────────────────
            selected: Dict[str, List[str]] = {}
            for omic in omics:
                X_tr = X_by_view[omic].iloc[train_idx]
                obs = ~X_tr.isna().all(axis=1)
                X_obs = X_tr.loc[obs]
                y_obs = y_train[obs.to_numpy()]
                all_cols = X_by_view[omic].columns.tolist()
                if use_ratio:
                    k = max(1, int(ratio * len(all_cols)))
                k_eff = min(k, len(all_cols))
                if X_obs.shape[0] < 5 or np.unique(y_obs).size < 2:
                    selected[omic] = all_cols[:k_eff]
                else:
                    selected[omic] = _anova_select(X_obs, y_obs, k_eff)

            # ── Scale per omic (fit on observed train patients) ───────────────
            scalers: Dict[str, Tuple[StandardScaler, pd.Series]] = {}
            for omic in omics:
                cols = selected[omic]
                X_tr_sel = X_by_view[omic].iloc[train_idx][cols]
                obs_tr = ~X_tr_sel.isna().all(axis=1)
                col_means = X_tr_sel.loc[obs_tr].mean()
                sc = StandardScaler()
                sc.fit(X_tr_sel.loc[obs_tr].fillna(col_means))
                scalers[omic] = (sc, col_means)

            def _apply_scale(omic: str, df: pd.DataFrame) -> pd.DataFrame:
                sc, col_means = scalers[omic]
                cols = selected[omic]
                filled = df[cols].fillna(col_means)
                return pd.DataFrame(
                    sc.transform(filled), index=df.index, columns=cols
                )

            # ── Build train-only omic views (observed patients per omic) ─────
            train_views: List[pd.DataFrame] = []
            train_mod_names: List[str] = []
            for omic in omics:
                X_tr = X_by_view[omic].iloc[train_idx]
                obs = ~X_tr.isna().all(axis=1)
                X_tr_obs = _apply_scale(omic, X_tr).loc[obs]
                if X_tr_obs.shape[0] > 0:
                    train_views.append(X_tr_obs)
                    train_mod_names.append(omic)

            if not train_views:
                continue

            # ── Build all-patient omic views for predictor ───────────────────
            all_views: List[pd.DataFrame] = []
            all_mod_names: List[str] = []
            for omic in omics:
                obs_all = ~X_by_view[omic].isna().all(axis=1)
                X_obs_all = _apply_scale(omic, X_by_view[omic]).loc[obs_all]
                if X_obs_all.shape[0] > 0:
                    all_views.append(X_obs_all)
                    all_mod_names.append(omic)

            # ── Train IntegrAO ────────────────────────────────────────────────
            eff_nb_train = min(self.neighbor_size, min(v.shape[0] for v in train_views) - 1)
            temp_dir = tempfile.mkdtemp(prefix=f"integrao_fold{fold_idx}_")
            try:
                integrater = integrao_integrater(
                    train_views,
                    dataset_name=experiment_spec.dataset_name,
                    modalities_name_list=train_mod_names,
                    neighbor_size=eff_nb_train,
                    embedding_dims=self.embedding_dims,
                    fusing_iteration=self.fusing_iteration,
                    normalization_factor=self.normalization_factor,
                    alighment_epochs=self.align_epochs,
                    mu=self.mu,
                )
                integrater.network_diffusion()
                embeds_train, _, uns_model = integrater.unsupervised_alignment()
                uns_path = os.path.join(temp_dir, "model.pth")
                torch.save(uns_model.state_dict(), uns_path)

                clf = None
                sup_path = None

                if self.mode == "supervised":
                    y_train_series = pd.Series(y_train, index=train_ids)
                    # reindex to integrater's patient order, drop any NaN
                    clf_labels = y_train_series.reindex(
                        list(integrater.dict_sampleToIndexs.keys())
                    ).dropna().astype(int)
                    clf_labels_df = pd.DataFrame({"label": clf_labels})

                    _, _, sup_model, _ = integrater.classification_finetuning(
                        clf_labels=clf_labels_df["label"],
                        model_path=temp_dir,
                        finetune_epochs=self.finetune_epochs,
                    )
                    sup_path = os.path.join(temp_dir, "model_supervised.pth")
                    torch.save(sup_model.state_dict(), sup_path)

                else:  # unsupervised
                    y_tr_s = pd.Series(y_train, index=train_ids)
                    y_emb = (
                        y_tr_s.reindex(embeds_train.index).dropna().astype(int)
                    )
                    clf = LogisticRegression(
                        C=self.lr_C,
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        multi_class="auto",
                    )
                    clf.fit(
                        embeds_train.loc[y_emb.index].to_numpy(),
                        y_emb.to_numpy(),
                    )

                t_fit = time.perf_counter() - t_start

                # ── Inference ──────────────────────────────────────────────────
                t1 = time.perf_counter()
                eff_nb_all = min(
                    self.neighbor_size, min(v.shape[0] for v in all_views) - 1
                )
                predictor = integrao_predictor(
                    all_views,
                    dataset_name=experiment_spec.dataset_name,
                    modalities_name_list=all_mod_names,
                    neighbor_size=eff_nb_all,
                    embedding_dims=self.embedding_dims,
                    hidden_channels=self.hidden_channels,
                    fusing_iteration=self.fusing_iteration,
                    normalization_factor=self.normalization_factor,
                    mu=self.mu,
                    num_classes=n_classes,
                )
                predictor.network_diffusion()

                # dict_sampleToIndexs keys give the insertion-order patient IDs
                pred_index = list(predictor.dict_sampleToIndexs.keys())
                fallback = int(
                    pd.Series(y_train).value_counts().idxmax()
                )

                if self.mode == "supervised":
                    raw_preds = predictor.inference_supervised(
                        sup_path,
                        new_datasets=all_views,
                        modalities_names=all_mod_names,
                    )
                    pred_s = pd.Series(raw_preds, index=pred_index)
                    y_pred = np.array(
                        [int(pred_s.get(pid, fallback)) for pid in test_ids]
                    )

                else:  # unsupervised
                    emb_df, _ = predictor.inference_unsupervised(
                        uns_path,
                        new_datasets=all_views,
                        modalities_names=all_mod_names,
                    )
                    zero_emb = np.zeros(self.embedding_dims)
                    test_emb = np.stack(
                        [
                            emb_df.loc[pid].to_numpy()
                            if pid in emb_df.index
                            else zero_emb
                            for pid in test_ids
                        ]
                    )
                    y_pred = clf.predict(test_emb)

                t_pred = time.perf_counter() - t1

                artifact = {
                    "model_type": "integrao",
                    "mode": self.mode,
                    "uns_state_dict": {
                        k: v.detach().cpu() for k, v in uns_model.state_dict().items()
                    },
                    "sup_state_dict": (
                        {k: v.detach().cpu() for k, v in sup_model.state_dict().items()}
                        if self.mode == "supervised"
                        else None
                    ),
                    "classifier": clf,
                    "scalers": {omic: scalers[omic] for omic in selected},
                    "selected": {omic: list(cols) for omic, cols in selected.items()},
                    "embedding_dims": int(self.embedding_dims),
                    "hidden_channels": int(self.hidden_channels),
                    "n_classes": int(n_classes),
                    "neighbor_size_train": int(eff_nb_train),
                    "fold": int(fold_idx),
                    "label_classes": le.classes_.tolist(),
                    "train_idx": [int(i) for i in train_idx],
                    "test_idx": [int(i) for i in test_idx],
                }
                self.fold_artifacts_.append(artifact)

            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

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
                        "mode": self.mode,
                        "neighbor_size": eff_nb_train,
                        "embedding_dims": self.embedding_dims,
                        "align_epochs": self.align_epochs,
                        "finetune_epochs": self.finetune_epochs
                        if self.mode == "supervised"
                        else None,
                        "k_per_omic": k if not use_ratio else None,
                        "ratio_per_omic": ratio if use_ratio else None,
                        "lr_C": self.lr_C if self.mode == "unsupervised" else None,
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
                "integration": "integrao",
                "mode": self.mode,
            },
        )
