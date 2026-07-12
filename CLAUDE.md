# Coordination — two users finishing the benchmark in parallel

This repo's SLURM jobs are being run by **two people at once** to finish faster.
The SLURM QOS submit cap (`MaxSubmitJobsPerUser=1801`) and fair-share priority are
**per user**, so two accounts = two independent lanes ≈ double throughput.

**All results land in ONE shared tree** (`<model>/results/...`). A run directory
exists **only if that config finished**. So the hard rule is:

> **Never run a config that the other account owns.** If both accounts submit the
> same config, they double-run into the same results tree — wasted compute and a
> race. The work is partitioned by *model* so overlap is impossible as long as each
> side runs only its own models.

## Division of labor

| Owner | Models | Submitter to launch |
|---|---|---|
| **Primary account (mstojano)** | `logreg_early` + all core reruns (`integrao`, `pnet`, `mofa`) | already running — do not touch |
| **SECOND user (you, if you are the friend)** | `svm_early`, `pca_logreg`, `logreg_late` | `sbatch slurm_submit_daemon_her.sh` |

## FIRST: which user are you? (run this — don't guess)

```bash
whoami
```
- If it prints **`mstojano`** → you are the PRIMARY account. Everything is already
  running; do nothing here.
- If it prints **anything else** → you are the **SECOND user (the friend)**. Follow
  the steps below. (You don't need the human to tell you — the username decides it.)

## If you are the SECOND user (the friend) — do exactly this

1. From this directory (`code/full-test-pipeline`), launch your durable submitter:
   ```bash
   sbatch slurm_submit_daemon_her.sh
   ```
   That starts job `submit-her`: a throttled, self-chaining SLURM-hosted submitter
   that feeds **only** `svm_early`, `pca_logreg`, `logreg_late` (6,858 jobs, all
   fast: ~2–10 min each). It keeps your queue just under the 1,801 cap and tops up
   every 90s until all are submitted. It survives the 72h walltime by re-submitting
   itself (`--signal USR1` trap). Resumable via `submit_daemon_her.state`.

2. Watch progress:
   ```bash
   tail -f submit_daemon_her.log          # submission progress (cumulative=…)
   squeue -u $USER                        # your running/pending array jobs
   ```

3. **Do NOT**:
   - run `svm/pca/late` any other way, or run `logreg_early`, `integrao`, `pnet`,
     `mofa`, or any `*_rerun` / `submit_daemon_rerun` / `submit_daemon_sk` script —
     those are the primary account's lane.
   - edit anything under `code/evaluation/` — shared framework, off-limits.
   - regenerate `experiments_*.txt` — they already exist and are correct.

4. If `sbatch` complains about account/QOS, add your default account:
   `sbatch --account=<your-account> slurm_submit_daemon_her.sh`. Nothing else changes.

## How to tell it's working

- `submit_daemon_her.log` shows lines like `submit svm_early ... cumulative=N`.
- `squeue -u $USER` fills with `svm_earl`, `pca_logr`, `logreg_l` array elements.
- Completed runs appear as new dirs with a `fold_models/` subdir under each model's
  `results/`. When the log prints `ALL HER MODELS SUBMITTED`, submission is done
  (jobs may still be running); when your queue drains, the work is finished.

## Everything else (splits/seeds/axes/HP grids) is already fixed and identical

Do not change model code, seeds (0/1/2), CV (5/3), axis values, or HP grids — the
benchmark's whole point is that every model uses identical splits. The model scripts
and `make_experiments.sh` already encode the standard. You only launch the submitter.
