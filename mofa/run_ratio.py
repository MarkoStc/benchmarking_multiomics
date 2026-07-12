"""One SLURM array task: fixed MOFA HPs, one feature-ratio per omic."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation"))
from results_util import save_benchmark_result
from common import build_cv_spec, build_ratio_fs_spec, f2tag, parse_omics
from mofa_model import MOFAModel, load_compat_dataset as load_dataset, dump_fold_artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--omics", required=True)
    p.add_argument("--n-latent", type=int, required=True)
    p.add_argument("--downstream-c", type=float, required=True)
    p.add_argument("--ratio-per-omic", type=float, required=True)
    p.add_argument("--include-non-intersection-frac", type=float, default=1.0)
    p.add_argument("--missing-policy", choices=["intersection", "impute"], default="impute")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--output-root", default="results_mofa_ratio")
    args = p.parse_args()

    omics = parse_omics(args.omics)
    dataset = load_dataset(args.dataset_path)
    cv_spec = build_cv_spec(args.outer_splits, inner_splits=3, random_state=args.random_state)
    fs_spec = build_ratio_fs_spec(ratio_per_omic=args.ratio_per_omic)

    model = MOFAModel(
        cv_spec=cv_spec,
        feature_selection_spec=fs_spec,
        n_latent=args.n_latent,
        downstream_c=args.downstream_c,
    )

    result = model.evaluate_on_feature_ratio(
        dataset=dataset,
        omics=omics,
        ratios=[args.ratio_per_omic],
        selection_method="anova",
        include_non_intersection_frac=args.include_non_intersection_frac,
        missing_policy=args.missing_policy,
        notes="MOFA fixed-HP feature-ratio sweep via Slurm array",
    )

    run_name = (
        f"mofa"
        f"__nl{args.n_latent}"
        f"__C{f2tag(args.downstream_c)}"
        f"__ratio{f2tag(args.ratio_per_omic)}"
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
                "mofa_model.py",
                "run_ratio.py",
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
