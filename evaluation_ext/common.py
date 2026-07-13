from __future__ import annotations

import os
from typing import List

import joblib

from base import CVSpec, FeatureSelectionSpec, MultiOmicsDataset


def load_dataset(dataset_path: str) -> MultiOmicsDataset:
    payload = joblib.load(dataset_path)
    return MultiOmicsDataset(
        name=payload["name"],
        views=payload["views"],
        y=payload["y"],
        patient_ids=payload["patient_ids"],
        metadata=payload.get("metadata", {}),
    )


def parse_omics(omics_csv: str) -> List[str]:
    return [x.strip() for x in omics_csv.split(",") if x.strip()]


def get_n_jobs_from_env(default: int = 1) -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", str(default)))


def build_cv_spec(outer_splits: int, inner_splits: int, random_state: int) -> CVSpec:
    return CVSpec(
        outer_splits=int(outer_splits),
        inner_splits=int(inner_splits),
        random_state=int(random_state),
        shuffle=True,
    )


def build_k_fs_spec(
    method: str = "anova",
    k_per_omic: int = 100,
    variance_threshold: float = 0.0,
) -> FeatureSelectionSpec:
    return FeatureSelectionSpec(
        method=method,
        k_per_omic=int(k_per_omic),
        variance_threshold=float(variance_threshold),
    )


def build_ratio_fs_spec(
    method: str = "anova",
    ratio_per_omic: float = 0.1,
    variance_threshold: float = 0.0,
) -> FeatureSelectionSpec:
    return FeatureSelectionSpec(
        method=method,
        ratio_per_omic=float(ratio_per_omic),
        variance_threshold=float(variance_threshold),
    )


def f2tag(x) -> str:
    return str(x).replace(".", "p")
