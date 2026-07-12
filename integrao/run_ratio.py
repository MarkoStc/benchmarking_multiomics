"""One SLURM array task: fixed IntegrAO HPs, one feature-ratio value."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../evaluation"))
from results_util import save_benchmark_result
from common import build_cv_spec, build_ratio_fs_spec, f2tag, load_dataset, parse_omics
from integrao_model import IntegrAOModel, dump_fold_artifacts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--omics", required=True)
    p.add_argument("--mode", choices=["supervised", "unsupervised"], required=True)
    p.add_argument("--ratio-per-omic", type=float, required=True)
    p.add_argument("--neighbor-size", type=int, required=True)
    p.add_argument("--embedding-dims", type=int, required=True)
    p.add_argument("--align-epochs", type=int, default=1000)
    p.add_argument("--finetune-epochs", type=int, default=1000)
    p.add_argument("--lr-C", type=float, default=1.0)
    p.add_argument("--include-non-intersection-frac", type=float, default=1.0)
    p.add_argument("--missing-policy", choices=["intersection", "impute"], default="impute")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--output-root", default="results_integrao_ratio")
    args = p.parse_args()

    omics = parse_omics(args.omics)
    dataset = load_dataset(args.dataset_path)
    cv_spec = build_cv_spec(args.outer_splits, inner_splits=3, random_state=args.random_state)
    fs_spec = build_ratio_fs_spec(ratio_per_omic=args.ratio_per_omic)

    model = IntegrAOModel(
        cv_spec=cv_spec,
        feature_selection_spec=fs_spec,
        mode=args.mode,
        neighbor_size=args.neighbor_size,
        embedding_dims=args.embedding_dims,
        align_epochs=args.align_epochs,
        finetune_epochs=args.finetune_epochs,
        lr_C=args.lr_C,
    )

    result = model.evaluate_on_feature_ratio(
        dataset=dataset,
        omics=omics,
        ratios=[args.ratio_per_omic],
        selection_method="anova",
        missing_policy=args.missing_policy,
        include_non_intersection_frac=args.include_non_intersection_frac,
        notes="IntegrAO fixed-HP feature-ratio sweep via Slurm array",
    )

    run_name = (
        f"integrao__{args.mode}"
        f"__nb{args.neighbor_size}"
        f"__emb{args.embedding_dims}"
        f"__ratio{f2tag(args.ratio_per_omic)}"
        + (f"__C{f2tag(args.lr_C)}" if args.mode == "unsupervised" else "")
        + f"__seed{args.random_state}"
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
                "integrao_model.py",
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
