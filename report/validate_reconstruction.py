"""Validate that reconstructed predictions reproduce each run's stored per-fold
accuracy & balanced_accuracy EXACTLY. If they match, all extended metrics derived
from the same reconstructed predictions are trustworthy.

Usage:
  python validate_reconstruction.py <model_dir> [n_runs_per_axis]
e.g.
  python validate_reconstruction.py logreg_early 2
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reconstruct import predict_run, reconstruct_cohort, load_metadata  # noqa: E402

ROOT = os.path.dirname(HERE)
AXES = ["npatients", "ratio", "nomics", "missing"]
TOL = 1e-9


def find_runs(model_dir: str, axis: str, k: int):
    base = os.path.join(ROOT, model_dir, "results")
    hits = []
    for dp, dns, fns in os.walk(base):
        if os.path.basename(dp) == "fold_models" and f"/{axis}/" in dp + "/":
            hits.append(os.path.dirname(dp))
    random.shuffle(hits)
    return hits[:k]


def validate_run(run_dir: str) -> dict:
    fr = pd.read_csv(os.path.join(run_dir, "fold_results.csv"))
    stored = {int(r.fold): (float(r.accuracy), float(r.balanced_accuracy)) for r in fr.itertuples()}

    # verify X row-order by reproducing the fold split from scratch
    meta = load_metadata(run_dir)
    ds, omics, cohort, mp = reconstruct_cohort(meta)
    y = ds.y.loc[cohort].to_numpy()
    seed = int(meta["extra_metadata"]["random_state"])
    ns = int(meta["extra_metadata"]["outer_splits"])
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
    fresh = [te.tolist() for _, te in skf.split(np.zeros(len(y)), y)]

    max_acc_err = 0.0
    max_bacc_err = 0.0
    split_ok = True
    n_folds = 0
    for f in predict_run(run_dir):
        n_folds += 1
        acc = accuracy_score(f["y_true"], f["y_pred"])
        bacc = balanced_accuracy_score(f["y_true"], f["y_pred"])
        s_acc, s_bacc = stored[f["fold"]]
        max_acc_err = max(max_acc_err, abs(acc - s_acc))
        max_bacc_err = max(max_bacc_err, abs(bacc - s_bacc))
        if f["test_idx"].tolist() != fresh[f["fold"] - 1]:
            split_ok = False
    return {
        "run": os.path.relpath(run_dir, ROOT),
        "n_folds": n_folds,
        "max_acc_err": max_acc_err,
        "max_bacc_err": max_bacc_err,
        "split_ok": split_ok,
        "ok": (max_acc_err < TOL and max_bacc_err < TOL and split_ok),
    }


def main():
    model_dir = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    random.seed(0)
    runs = []
    for axis in AXES:
        runs += find_runs(model_dir, axis, k)
    print(f"### {model_dir}: validating {len(runs)} runs")
    all_ok = True
    for rd in runs:
        try:
            r = validate_run(rd)
        except Exception as e:
            import traceback
            print(f"  ERROR {os.path.relpath(rd, ROOT)}: {type(e).__name__}: {e}")
            traceback.print_exc()
            all_ok = False
            continue
        flag = "OK " if r["ok"] else "FAIL"
        print(f"  [{flag}] folds={r['n_folds']} acc_err={r['max_acc_err']:.2e} "
              f"bacc_err={r['max_bacc_err']:.2e} split_ok={r['split_ok']} | {r['run']}")
        all_ok = all_ok and r["ok"]
    print(f"### {model_dir}: {'ALL MATCH' if all_ok else 'MISMATCHES PRESENT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
