"""Stage-1 harvester: for one (model, dataset, axis), walk every completed run,
reconstruct held-out predictions, compute the full metric battery, and write
three canonical parquet shards:

  canonical/fold_metrics/<model>__<dataset>__<axis>.parquet   (1 row / fold)
  canonical/per_class/<model>__<dataset>__<axis>.parquet      (1 row / fold / class)
  canonical/predictions/<model>__<dataset>__<axis>.parquet    (1 row / test patient)

Usage: python harvest.py <model> <dataset> <axis>
  e.g. python harvest.py logreg_early TCGA-KIPAN npatients
"""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from reconstruct import predict_run, load_metadata  # noqa: E402
from metrics import fold_metrics, per_class_metrics  # noqa: E402

_COMMON = {"dataset_path", "output_root", "omics", "random_state", "n_patients",
           "include_non_intersection_frac", "missing_policy", "outer_splits"}


def condition_cols(meta: dict) -> dict:
    em = meta["extra_metadata"]
    omics = [o.strip() for o in em["omics"].split(",") if o.strip()]
    integration = meta["model"].get("integration", "")
    hparams = {k: v for k, v in em.items() if k not in _COMMON}
    return {
        "integration": integration,
        "omic_combination": "+".join(omics),
        "n_omics": len(omics),
        "n_patients": em.get("n_patients"),
        "include_non_intersection_frac": em.get("include_non_intersection_frac"),
        "ratio_per_omic": em.get("ratio_per_omic"),
        "k_per_omic": em.get("k_per_omic"),
        "seed": int(em["random_state"]),
        "hparams": json.dumps(hparams, default=str, sort_keys=True),
    }


def find_run_dirs(model, dataset, axis):
    base = os.path.join(ROOT, model, "results", dataset, axis)
    hits = []
    if not os.path.isdir(base):
        return hits
    for dp, dns, fns in os.walk(base):
        if os.path.basename(dp) == "fold_models":
            hits.append(os.path.dirname(dp))
    return sorted(hits)


def harvest(model, dataset, axis):
    runs = find_run_dirs(model, dataset, axis)
    fm_rows, pc_rows, pred_rows = [], [], []
    n_ok = n_err = 0
    for rd in runs:
        try:
            meta = load_metadata(rd)
            cond = condition_cols(meta)
            fr = pd.read_csv(os.path.join(rd, "fold_results.csv"))
            rt = {int(r.fold): float(r.total_time_sec) for r in fr.itertuples()}
            base = {"dataset": dataset, "model": model, "axis": axis,
                    "run_name": meta["run_name"], **cond}
            for f in predict_run(rd):
                fold = int(f["fold"])
                m = fold_metrics(f["y_true"], f["y_pred"], f["proba"], f["classes"])
                fm_rows.append({**base, "fold": fold, "n_test": len(f["y_true"]),
                                "runtime_seconds": rt.get(fold), **m})
                for pc in per_class_metrics(f["y_true"], f["y_pred"], f["proba"], f["classes"]):
                    pc_rows.append({**base, "fold": fold, **pc})
                proba = f["proba"]
                classes = np.asarray(f["classes"])
                for j, pid in enumerate(f["patient_ids"]):
                    row = {**base, "fold": fold, "patient_id": pid,
                           "true_label": f["y_true"][j], "pred_label": f["y_pred"][j]}
                    if proba is not None:
                        for ci, c in enumerate(classes.tolist()):
                            row[f"proba_{c}"] = float(np.asarray(proba)[j, ci])
                    pred_rows.append(row)
            n_ok += 1
        except Exception:
            n_err += 1
            sys.stderr.write(f"ERR {rd}\n")
            traceback.print_exc()
    tag = f"{model}__{dataset}__{axis}"
    for kind, rows in [("fold_metrics", fm_rows), ("per_class", pc_rows), ("predictions", pred_rows)]:
        d = os.path.join(HERE, "canonical", kind)
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, f"{tag}.pkl.gz")
        pd.DataFrame(rows).to_pickle(out, compression="gzip")
    print(f"{tag}: runs_ok={n_ok} runs_err={n_err} folds={len(fm_rows)} "
          f"preds={len(pred_rows)} perclass={len(pc_rows)}")
    return n_err


if __name__ == "__main__":
    model, dataset, axis = sys.argv[1], sys.argv[2], sys.argv[3]
    rc = harvest(model, dataset, axis)
    sys.exit(1 if rc else 0)
