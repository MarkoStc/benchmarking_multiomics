"""MOFA one-omic-at-a-time feature sweep (README 9.3): target omic at `ratio`,
all other omics at 100%. Uses ../evaluation_ext."""
from __future__ import annotations
import argparse, os, sys

_EXT = os.path.join(os.path.dirname(__file__), "../evaluation_ext")
sys.path.insert(0, _EXT)
import base, common, results_util  # noqa: F401,E402
from results_util import save_benchmark_result  # noqa: E402
from common import build_cv_spec, build_k_fs_spec, f2tag, parse_omics  # noqa: E402
from mofa_model import MOFAModel, load_compat_dataset as load_dataset, dump_fold_artifacts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--omics", required=True)
    p.add_argument("--target-omic", required=True)
    p.add_argument("--n-latent", type=int, required=True)
    p.add_argument("--downstream-c", type=float, required=True)
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--include-non-intersection-frac", type=float, default=1.0)
    p.add_argument("--missing-policy", choices=["intersection", "impute"], default="impute")
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--output-root", default="results/feature_peromic")
    args = p.parse_args()

    omics = parse_omics(args.omics)
    dataset = load_dataset(args.dataset_path)
    cv_spec = build_cv_spec(args.outer_splits, inner_splits=3, random_state=args.random_state)
    model = MOFAModel(cv_spec=cv_spec, feature_selection_spec=build_k_fs_spec(k_per_omic=100),
                      n_latent=args.n_latent, downstream_c=args.downstream_c)

    result = model.evaluate_on_feature_ratio_one_omic(
        dataset=dataset, omics=omics, target_omic=args.target_omic, ratios=[args.ratio],
        selection_method="anova", include_non_intersection_frac=args.include_non_intersection_frac,
        missing_policy=args.missing_policy,
        notes="MOFA one-omic-at-a-time feature sweep (README 9.3) via Slurm array")

    run_name = (f"mofa__nl{args.n_latent}__C{f2tag(args.downstream_c)}__omic{args.target_omic}"
                f"__ratio{f2tag(args.ratio)}__seed{args.random_state}")
    run_dir = save_benchmark_result(
        result=result, output_root=args.output_root, run_name=run_name, model=model,
        dataset=dataset, selected_omics=omics,
        source_files=[os.path.join(os.path.dirname(__file__), "mofa_model.py"),
                      os.path.join(os.path.dirname(__file__), "run_feature_peromic.py"),
                      os.path.join(_EXT, "base.py")],
        extra_metadata=vars(args))
    dump_fold_artifacts(model, run_dir)
    print(f"Saved results to: {run_dir}")


if __name__ == "__main__":
    main()
