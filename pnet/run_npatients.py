"""One SLURM array task: fixed P-NET HPs, one patient-count value."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation"))
from results_util import save_benchmark_result
from common import build_cv_spec, build_k_fs_spec, f2tag, load_dataset, parse_omics
from pnet_model import PNetModel, dump_fold_artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--omics", required=True)
    p.add_argument("--hidden-units", type=int, required=True)
    p.add_argument("--dropout", type=float, required=True)
    p.add_argument("--w-reg", type=float, required=True)
    p.add_argument("--k-per-omic", type=int, required=True)
    p.add_argument("--n-patients", type=int, required=True)
    p.add_argument("--include-non-intersection-frac", type=float, default=1.0)
    p.add_argument("--missing-policy", choices=["intersection", "impute"], default="impute")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--output-root", default="results_pnet_npatients")
    args = p.parse_args()

    omics = parse_omics(args.omics)
    dataset = load_dataset(args.dataset_path)
    cv_spec = build_cv_spec(args.outer_splits, inner_splits=3, random_state=args.random_state)
    fs_spec = build_k_fs_spec(k_per_omic=args.k_per_omic)

    model = PNetModel(
        cv_spec=cv_spec,
        feature_selection_spec=fs_spec,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        w_reg=args.w_reg,
        epochs=args.epochs,
    )

    result = model.evaluate_on_n_patients(
        dataset=dataset,
        omics=omics,
        n_values=[args.n_patients],
        missing_policy=args.missing_policy,
        include_non_intersection_frac=args.include_non_intersection_frac,
        feature_selection_spec=fs_spec,
        notes="P-NET fixed-HP n-patients sweep via Slurm array",
    )

    run_name = (
        f"pnet"
        f"__hu{args.hidden_units}"
        f"__do{f2tag(args.dropout)}"
        f"__wr{f2tag(args.w_reg)}"
        f"__k{args.k_per_omic}"
        f"__n{args.n_patients}"
        f"__seed{args.random_state}"
    )

    run_dir = save_benchmark_result(
        result=result,
        output_root=args.output_root,
        run_name=run_name,
        model=model,
        dataset=dataset,
        selected_omics=omics,
        source_files=[
            os.path.join(os.path.dirname(__file__), f)
            for f in [
                "pnet_model.py",
                "run_npatients.py",
                "../../evaluation/base.py",
                "../../evaluation/results_util.py",
                "../../evaluation/common.py",
            ]
        ],
        extra_metadata=vars(args),
    )
    dump_fold_artifacts(model, run_dir)
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
