from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd


def _json_ready(obj: Any) -> Any:
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _json_ready(obj.to_dict())
        except Exception:
            pass

    if is_dataclass(obj):
        return _json_ready(asdict(obj))

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, pd.DataFrame):
        return {
            "__kind__": "DataFrame",
            "columns": list(obj.columns),
            "index": list(obj.index),
            "data": obj.to_dict(orient="records"),
        }

    if isinstance(obj, pd.Series):
        return {
            "__kind__": "Series",
            "name": obj.name,
            "index": list(obj.index),
            "data": obj.to_dict(),
        }

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_ready(v) for v in obj]

    return obj


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(_json_ready(obj), ensure_ascii=False, sort_keys=True)


def _is_scalar_like(x: Any) -> bool:
    return isinstance(
        x,
        (str, int, float, bool, type(None), np.integer, np.floating, np.bool_),
    )


def _extract_model_config(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}

    keys = [
        "cv_spec",
        "feature_selection_spec",
        "fusion_spec",
        "integration",
        "n_jobs",
        "top_n_interpret_features",
        "store_interpretability_per_fold",
        "probability",
        "max_iter",
    ]
    cfg = {"class_name": model.__class__.__name__}
    for key in keys:
        if hasattr(model, key):
            cfg[key] = getattr(model, key)
    return _json_ready(cfg)


def _extract_dataset_info(
    dataset: Any,
    selected_omics: Optional[Sequence[str]],
) -> Dict[str, Any]:
    if dataset is None:
        return {}

    info = {
        "name": getattr(dataset, "name", None),
        "n_patients": len(getattr(dataset, "patient_ids", []))
        if hasattr(dataset, "patient_ids")
        else None,
        "omics_available": list(getattr(dataset, "omics", []))
        if hasattr(dataset, "omics")
        else None,
    }

    if selected_omics is not None:
        info["selected_omics"] = list(selected_omics)

    if hasattr(dataset, "summary"):
        try:
            info["summary"] = _json_ready(dataset.summary(selected_omics=selected_omics))
        except TypeError:
            info["summary"] = _json_ready(dataset.summary())

    return info


def _summary_frame_for_result(result: Any) -> pd.DataFrame:
    if hasattr(result, "summary_frame") and callable(result.summary_frame):
        return result.summary_frame().copy()

    if hasattr(result, "folds"):
        metadata = getattr(result, "metadata", {}) or {}
        row = {
            "accuracy_mean": getattr(result, "accuracy_mean", None),
            "accuracy_std": getattr(result, "accuracy_std", None),
            "balanced_accuracy_mean": getattr(result, "balanced_accuracy_mean", None),
            "balanced_accuracy_std": getattr(result, "balanced_accuracy_std", None),
            "total_time_mean_sec": getattr(result, "total_time_mean_sec", None),
            "total_time_std_sec": getattr(result, "total_time_std_sec", None),
            "n_folds": len(getattr(result, "folds", [])),
        }
        for k, v in metadata.items():
            if _is_scalar_like(v):
                row[f"meta__{k}"] = v
        return pd.DataFrame([row])

    return pd.DataFrame()


def _fold_row(
    fold: Any,
    axis_name: Optional[str] = None,
    axis_value: Any = None,
    experiment_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "fold": getattr(fold, "fold", None),
        "n_train": getattr(fold, "n_train", None),
        "n_test": getattr(fold, "n_test", None),
        "accuracy": getattr(fold, "accuracy", None),
        "balanced_accuracy": getattr(fold, "balanced_accuracy", None),
        "fit_time_sec": getattr(fold, "fit_time_sec", None),
        "predict_time_sec": getattr(fold, "predict_time_sec", None),
        "total_time_sec": getattr(fold, "total_time_sec", None),
        "best_params_json": _safe_json_dumps(getattr(fold, "best_params", {})),
    }

    interp = getattr(fold, "interpretability", None)
    if interp is not None:
        row["interpretability_supported"] = getattr(interp, "supported", None)
        row["interpretability_method"] = getattr(interp, "method", None)
        row["interpretability_notes"] = getattr(interp, "notes", None)
    else:
        row["interpretability_supported"] = None
        row["interpretability_method"] = None
        row["interpretability_notes"] = None

    if axis_name is not None:
        row[axis_name] = axis_value if _is_scalar_like(axis_value) else repr(axis_value)

    for k, v in (experiment_metadata or {}).items():
        if _is_scalar_like(v):
            row[f"meta__{k}"] = v

    return row


def _folds_frame_for_result(result: Any) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if hasattr(result, "results") and hasattr(result, "axis_name"):
        axis_name = getattr(result, "axis_name")
        axis_values = list(getattr(result, "axis_values", []))
        result_map = getattr(result, "results")

        for axis_value in axis_values:
            exp = result_map[axis_value]
            for fold in getattr(exp, "folds", []):
                rows.append(
                    _fold_row(
                        fold,
                        axis_name=axis_name,
                        axis_value=axis_value,
                        experiment_metadata=getattr(exp, "metadata", {}),
                    )
                )
        return pd.DataFrame(rows)

    if hasattr(result, "folds"):
        for fold in getattr(result, "folds", []):
            rows.append(
                _fold_row(
                    fold,
                    experiment_metadata=getattr(result, "metadata", {}),
                )
            )
        return pd.DataFrame(rows)

    return pd.DataFrame()


def _collect_interpretability(result: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if hasattr(result, "results") and hasattr(result, "axis_name"):
        axis_name = getattr(result, "axis_name")
        for axis_value in getattr(result, "axis_values", []):
            exp = result.results[axis_value]
            per_fold = {}
            for fold in getattr(exp, "folds", []):
                interp = getattr(fold, "interpretability", None)
                if interp is not None:
                    per_fold[str(getattr(fold, "fold", len(per_fold)))] = _json_ready(
                        interp
                    )
            if per_fold:
                out[f"{axis_name}={axis_value}"] = per_fold
        return out

    if hasattr(result, "folds"):
        for fold in getattr(result, "folds", []):
            interp = getattr(fold, "interpretability", None)
            if interp is not None:
                out[str(getattr(fold, "fold", len(out)))] = _json_ready(interp)

    return out


def _save_artifact(obj: Any, path_base: Path) -> None:
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path_base.with_suffix(".csv"), index=False)
        return

    if isinstance(obj, pd.Series):
        obj.to_csv(path_base.with_suffix(".csv"), header=True)
        return

    if isinstance(obj, (dict, list, tuple)):
        with open(path_base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(_json_ready(obj), f, indent=2, ensure_ascii=False)
        return

    joblib.dump(obj, path_base.with_suffix(".joblib"))


def save_benchmark_result(
    result: Any,
    output_root: str | Path = "results",
    run_name: Optional[str] = None,
    model: Any = None,
    dataset: Any = None,
    selected_omics: Optional[Sequence[str]] = None,
    source_files: Optional[Sequence[str | Path]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    extra_artifacts: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a benchmark result (GridResult or ExperimentResult) into one reproducible run folder.

    Saved files:
      - metadata.json
      - result_object.joblib
      - result.json
      - summary.csv
      - fold_results.csv
      - interpretability.json (only if available)
      - artifacts/...
      - source_code/...
      - README.txt
    """
    output_root = Path(output_root)

    model_name = model.__class__.__name__ if model is not None else "unknown_model"
    dataset_name = (
        getattr(dataset, "name", "unknown_dataset")
        if dataset is not None
        else "unknown_dataset"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    run_stem = run_name or getattr(result, "axis_name", None) or "benchmark_run"

    run_dir = output_root / dataset_name / model_name / f"{run_stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_stem,
        "result_type": type(result).__name__,
        "model": _extract_model_config(model),
        "dataset": _extract_dataset_info(dataset, selected_omics=selected_omics),
        "extra_metadata": _json_ready(extra_metadata or {}),
    }

    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    joblib.dump(result, run_dir / "result_object.joblib")

    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(result), f, indent=2, ensure_ascii=False)

    summary_df = _summary_frame_for_result(result)
    if not summary_df.empty:
        summary_df.to_csv(run_dir / "summary.csv", index=False)

    folds_df = _folds_frame_for_result(result)
    if not folds_df.empty:
        folds_df.to_csv(run_dir / "fold_results.csv", index=False)

    interpretability = _collect_interpretability(result)
    if interpretability:
        with open(run_dir / "interpretability.json", "w", encoding="utf-8") as f:
            json.dump(interpretability, f, indent=2, ensure_ascii=False)

    if extra_artifacts:
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for name, obj in extra_artifacts.items():
            _save_artifact(obj, artifacts_dir / name)

    if source_files:
        code_dir = run_dir / "source_code"
        code_dir.mkdir(exist_ok=True)
        for src in source_files:
            src_path = Path(src)
            if src_path.exists() and src_path.is_file():
                shutil.copy2(src_path, code_dir / src_path.name)

    readme = f"""Saved benchmark run
==================

Folder: {run_dir}

Files
-----
- metadata.json: run/model/dataset metadata
- result_object.joblib: Python object with the full result
- result.json: JSON export of the structured result
- summary.csv: one-row or per-axis summary table
- fold_results.csv: per-fold metrics and best hyperparameters
- interpretability.json: saved if fold interpretability exists
- artifacts/: extra saved objects supplied by the notebook
- source_code/: copied source files used for the run
"""
    (run_dir / "README.txt").write_text(readme, encoding="utf-8")

    return run_dir