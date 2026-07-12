"""One SLURM array task: fixed LogRegEarlyModel HPs, one ratio value."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../sklearn_common"))
from results_util import save_benchmark_result
from common import build_cv_spec, build_ratio_fs_spec, f2tag, load_dataset, parse_omics
from sk_bench import LogRegEarlyModel as Model, dump_fold_artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--omics", required=True)
    p.add_argument("--downstream-c", type=float, required=True)
    p.add_argument("--ratio-per-omic", type=float, required=True)
    p.add_argument("--include-non-intersection-frac", type=float, default=1.0)
    p.add_argument("--missing-policy", choices=["intersection", "impute"], default="impute")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--output-root", default="results_logreg_early_ratio")
    args = p.parse_args()

    omics = parse_omics(args.omics)
    dataset = load_dataset(args.dataset_path)
    cv_spec = build_cv_spec(args.outer_splits, inner_splits=3, random_state=args.random_state)
    fs_spec = build_ratio_fs_spec(ratio_per_omic=args.ratio_per_omic)

    model = Model(
        cv_spec=cv_spec,
        feature_selection_spec=fs_spec,
        downstream_c=args.downstream_c,
    )

    result = model.evaluate_on_feature_ratio(
        dataset=dataset,
        omics=omics,
        ratios=[args.ratio_per_omic],
        selection_method="anova",
        include_non_intersection_frac=args.include_non_intersection_frac,
        missing_policy=args.missing_policy,
        notes="logreg_early fixed-HP feature-ratio sweep via Slurm array",
    )

    run_name = (
        f"logreg_early"
        f"__C{f2tag(args.downstream_c)}"
        + f"__ratio{f2tag(args.ratio_per_omic)}__seed{args.random_state}"
    )

    run_dir = save_benchmark_result(
        result=result,
        output_root=args.output_root,
        run_name=run_name,
        model=model,
        dataset=dataset,
        selected_omics=omics,
        source_files=[
            os.path.join(os.path.dirname(__file__), "run_ratio.py"),
            os.path.join(os.path.dirname(__file__), "../sklearn_common/sk_bench.py"),
            os.path.join(os.path.dirname(__file__), "../../evaluation/base.py"),
            os.path.join(os.path.dirname(__file__), "../../evaluation/results_util.py"),
            os.path.join(os.path.dirname(__file__), "../../evaluation/common.py"),
        ],
        extra_metadata=vars(args),
    )
    dump_fold_artifacts(model, run_dir)
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
