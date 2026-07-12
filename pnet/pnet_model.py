from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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


_REACTOME_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../reactome_files/reactome_gene_pathway_hsa_filtered.csv",
)

_reactome_df: Optional[pd.DataFrame] = None


def _load_reactome() -> pd.DataFrame:
    global _reactome_df
    if _reactome_df is None:
        _reactome_df = pd.read_csv(_REACTOME_PATH)
    return _reactome_df


def _strip_version(ensembl_id: str) -> str:
    return ensembl_id.split(".")[0]


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


class MaskedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("mask", mask.t().float())  # stored as (out, in)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def masked_weight(self) -> torch.Tensor:
        return self.weight * self.mask


class PNetTorch(nn.Module):
    def __init__(
        self,
        n_gene_features: int,
        n_genes: int,
        n_pathways: int,
        feature_gene_mask: torch.Tensor,
        gene_pathway_mask: torch.Tensor,
        n_dense_features: int,
        hidden_units: int,
        n_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.has_gene = n_gene_features > 0 and n_genes > 0 and n_pathways > 0
        self.has_dense = n_dense_features > 0

        if self.has_gene:
            self.feat_to_gene = MaskedLinear(n_gene_features, n_genes, feature_gene_mask)
            self.gene_to_path = MaskedLinear(n_genes, n_pathways, gene_pathway_mask)
            self.gene_bn = nn.BatchNorm1d(n_genes)
            self.path_bn = nn.BatchNorm1d(n_pathways)
            self.gene_drop = nn.Dropout(dropout)
            self.path_drop = nn.Dropout(dropout)

        if self.has_dense:
            self.dense_fc = nn.Linear(n_dense_features, hidden_units)
            self.dense_bn = nn.BatchNorm1d(hidden_units)
            self.dense_drop = nn.Dropout(dropout)

        head_in = (n_pathways if self.has_gene else 0) + (hidden_units if self.has_dense else 0)
        if head_in == 0:
            head_in = n_dense_features if n_dense_features > 0 else n_gene_features
        self.classifier = nn.Linear(head_in, n_classes)

    def forward(self, x_gene: Optional[torch.Tensor], x_dense: Optional[torch.Tensor]) -> torch.Tensor:
        parts = []

        if self.has_gene and x_gene is not None:
            h = torch.tanh(self.gene_drop(self.gene_bn(self.feat_to_gene(x_gene))))
            h = torch.tanh(self.path_drop(self.path_bn(self.gene_to_path(h))))
            parts.append(h)

        if self.has_dense and x_dense is not None:
            h = F.relu(self.dense_drop(self.dense_bn(self.dense_fc(x_dense))))
            parts.append(h)

        out = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.classifier(out)

    def l2_masked_penalty(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.classifier.weight.device)
        if self.has_gene:
            loss = loss + (self.feat_to_gene.masked_weight() ** 2).sum()
            loss = loss + (self.gene_to_path.masked_weight() ** 2).sum()
        return loss


def _build_masks(
    gene_cols: List[str],
    reactome_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    genes = [_strip_version(c.split("__", 1)[1]) for c in gene_cols]
    unique_genes = list(dict.fromkeys(genes))

    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    n_feats = len(gene_cols)
    n_genes = len(unique_genes)

    feature_gene_mask = torch.zeros(n_feats, n_genes)
    for i, g in enumerate(genes):
        j = gene_to_idx[g]
        feature_gene_mask[i, j] = 1.0

    sub = reactome_df[reactome_df["gene_id"].isin(set(unique_genes))]
    unique_pathways = list(dict.fromkeys(sub["pathway_id"].tolist()))
    pathway_to_idx = {p: i for i, p in enumerate(unique_pathways)}
    n_pathways = len(unique_pathways)

    gene_pathway_mask = torch.zeros(n_genes, n_pathways)
    for _, row in sub.iterrows():
        gi = gene_to_idx.get(row["gene_id"])
        pi = pathway_to_idx.get(row["pathway_id"])
        if gi is not None and pi is not None:
            gene_pathway_mask[gi, pi] = 1.0

    return feature_gene_mask, gene_pathway_mask, unique_genes, unique_pathways


class PNetModel(BaseModel):
    def __init__(
        self,
        cv_spec: CVSpec = CVSpec(),
        feature_selection_spec: FeatureSelectionSpec = FeatureSelectionSpec(
            method="anova", k_per_omic=100
        ),
        hidden_units: int = 64,
        dropout: float = 0.5,
        w_reg: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        n_jobs: int = 1,
    ):
        super().__init__(
            cv_spec=cv_spec,
            feature_selection_spec=feature_selection_spec,
            n_jobs=n_jobs,
            integration="early",
        )
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.w_reg = w_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def build_estimator(self):
        raise NotImplementedError("PNetModel uses its own training loop.")

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
    ) -> Tuple[np.ndarray, float, float]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reactome_df = _load_reactome()

        # Feature selection (ANOVA on train)
        if fs_spec.k_per_omic is not None:
            use_ratio = False
            k_base = int(fs_spec.k_per_omic)
        elif fs_spec.ratio_per_omic is not None:
            use_ratio = True
            ratio = float(fs_spec.ratio_per_omic)
        else:
            use_ratio = False
            k_base = 200

        selected_cols: List[str] = []
        for omic, cols in feature_groups.items():
            cols = [c for c in cols if c in X_train_df.columns]
            if not cols:
                continue
            X_tr_omic = X_train_df[cols]
            obs = ~X_tr_omic.isna().all(axis=1)
            X_obs = X_tr_omic.loc[obs]
            y_obs = y_train[obs.to_numpy()]
            if use_ratio:
                k = max(1, int(ratio * len(cols)))
            else:
                k = k_base
            k = min(k, len(cols))
            if X_obs.shape[0] < 5 or np.unique(y_obs).size < 2:
                selected_cols.extend(cols[:k])
            else:
                selected_cols.extend(_anova_select(X_obs, y_obs, k))

        if not selected_cols:
            selected_cols = X_train_df.columns.tolist()

        X_train_sel = X_train_df[selected_cols]
        X_test_sel = X_test_df[selected_cols]

        # Impute & scale
        col_means = X_train_sel.mean()
        X_train_filled = X_train_sel.fillna(col_means)
        X_test_filled = X_test_sel.fillna(col_means)

        sc = StandardScaler()
        X_train_np = sc.fit_transform(X_train_filled.to_numpy(dtype=np.float32))
        X_test_np = sc.transform(X_test_filled.to_numpy(dtype=np.float32))

        # Split gene vs dense features
        gene_mask_cols = [
            c for c in selected_cols
            if c.startswith("mrna__ENSG") or c.startswith("cnv__ENSG")
        ]
        dense_cols = [c for c in selected_cols if c not in set(gene_mask_cols)]

        gene_idx = [selected_cols.index(c) for c in gene_mask_cols]
        dense_idx = [selected_cols.index(c) for c in dense_cols]

        n_gene_features = len(gene_idx)
        n_dense_features = len(dense_idx)

        # Build masks
        feature_gene_mask = torch.zeros(1)
        gene_pathway_mask = torch.zeros(1)
        n_genes = 0
        n_pathways = 0

        if n_gene_features > 0:
            fgm, gpm, _, _ = _build_masks(gene_mask_cols, reactome_df)
            feature_gene_mask = fgm
            gene_pathway_mask = gpm
            n_genes = fgm.shape[1]
            n_pathways = gpm.shape[1]

        if n_gene_features == 0 or n_genes == 0 or n_pathways == 0:
            n_gene_features = 0
            n_genes = 0
            n_pathways = 0
            dense_idx = list(range(len(selected_cols)))
            n_dense_features = len(dense_idx)

        n_classes = int(np.unique(y_train).size)

        model = PNetTorch(
            n_gene_features=n_gene_features,
            n_genes=n_genes,
            n_pathways=n_pathways,
            feature_gene_mask=feature_gene_mask,
            gene_pathway_mask=gene_pathway_mask,
            n_dense_features=n_dense_features,
            hidden_units=self.hidden_units,
            n_classes=n_classes,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_tr_t = torch.from_numpy(X_train_np).float()
        y_tr_t = torch.from_numpy(y_train.astype(np.int64))
        X_te_t = torch.from_numpy(X_test_np).float()

        def _split(X: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
            xg = X[:, gene_idx].to(device) if n_gene_features > 0 else None
            xd = X[:, dense_idx].to(device) if n_dense_features > 0 else None
            return xg, xd

        # Train/val split for early stopping (10% val)
        n_tr = len(X_tr_t)
        n_val = max(1, int(0.1 * n_tr))
        n_fit = n_tr - n_val
        X_fit, X_val = X_tr_t[:n_fit], X_tr_t[n_fit:]
        y_fit, y_val = y_tr_t[:n_fit], y_tr_t[n_fit:]

        dataset_tr = TensorDataset(X_fit, y_fit)
        loader = DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience = 10
        no_improve = 0

        t0 = time.perf_counter()
        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                xg, xd = _split(xb)
                optimizer.zero_grad()
                logits = model(xg, xd)
                loss = criterion(logits, yb.to(device))
                loss = loss + self.w_reg * model.l2_masked_penalty()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                xgv, xdv = _split(X_val)
                val_logits = model(xgv, xdv)
                val_loss = criterion(val_logits, y_val.to(device)).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        t_fit = time.perf_counter() - t0

        model.load_state_dict(best_state)
        model.eval()
        t1 = time.perf_counter()
        with torch.no_grad():
            xg_te, xd_te = _split(X_te_t)
            logits = model(xg_te, xd_te)
            y_pred = logits.argmax(dim=1).cpu().numpy()
        t_pred = time.perf_counter() - t1

        artifact = {
            "model_type": "pnet",
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arch": {
                "n_gene_features": int(n_gene_features),
                "n_genes": int(n_genes),
                "n_pathways": int(n_pathways),
                "n_dense_features": int(n_dense_features),
                "hidden_units": int(self.hidden_units),
                "n_classes": int(n_classes),
                "dropout": float(self.dropout),
            },
            "feature_gene_mask": feature_gene_mask.detach().cpu(),
            "gene_pathway_mask": gene_pathway_mask.detach().cpu(),
            "selected_cols": list(selected_cols),
            "gene_mask_cols": list(gene_mask_cols),
            "gene_idx": list(gene_idx),
            "dense_idx": list(dense_idx),
            "scaler": sc,
            "col_means": col_means,
        }

        return y_pred, t_fit, t_pred, artifact

    def _nested_cv_evaluate(
        self,
        data,
        experiment_spec: ExperimentSpec,
        random_state: int = 0,
    ) -> ExperimentResult:
        if not isinstance(data, TabularPreparedData):
            raise TypeError("PNetModel requires TabularPreparedData.")

        X = data.X
        y = data.y
        feature_groups = data.feature_groups
        fs_spec = experiment_spec.feature_selection

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        outer_cv = StratifiedKFold(
            n_splits=experiment_spec.cv.outer_splits,
            shuffle=True,
            random_state=experiment_spec.cv.random_state,
        )

        folds: List[FoldResult] = []
        self.fold_artifacts_ = []
        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y_enc), start=1
        ):
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
                        "hidden_units": self.hidden_units,
                        "dropout": self.dropout,
                        "w_reg": self.w_reg,
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "lr": self.lr,
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
                "integration": "pnet",
            },
        )
