#!/bin/bash -l
#SBATCH --job-name=submit-her
#SBATCH --partition=academic
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/submit-her-%j.out
#SBATCH --error=logs/submit-her-%j.err

# Durable SLURM-hosted runner for the SECOND user's submitter (svm_early,
# pca_logreg, logreg_late). Self-chains on USR1 ~180s before walltime so
# submission survives the 72h limit. Runs under whoever sbatch's it.
set -u
ROOT=/work/scitas-share/FellayMultiOmic/code/full-test-pipeline
cd "$ROOT"

requeue() {
  echo "$(date) walltime approaching — resubmitting submit-her to continue"
  sbatch "$ROOT/slurm_submit_daemon_her.sh"
  exit 0
}
trap requeue USR1

echo "$(date) submit-her SLURM job $SLURM_JOB_ID starting"
bash "$ROOT/submit_daemon_her.sh" >> "$ROOT/submit_daemon_her.log" 2>&1 &
wait $!
echo "$(date) submit-her finished (ALL HER MODELS SUBMITTED)"
