#!/bin/bash -l
#SBATCH --job-name=submit-blockmiss
#SBATCH --partition=academic
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/submit-blockmiss-%j.out
#SBATCH --error=logs/submit-blockmiss-%j.err

# Durable SLURM-hosted runner for the block-missingness (README 8.2) submitter.
# Self-chains on USR1 ~180s before walltime so submission survives the 72h limit.
set -u
ROOT=/work/scitas-share/FellayMultiOmic/code/full-test-pipeline
cd "$ROOT"

requeue() {
  echo "$(date) walltime approaching — resubmitting submit-blockmiss to continue"
  sbatch "$ROOT/slurm_submit_daemon_blockmissing.sh"
  exit 0
}
trap requeue USR1

echo "$(date) submit-blockmiss SLURM job $SLURM_JOB_ID starting"
bash "$ROOT/submit_daemon_blockmissing.sh" >> "$ROOT/submit_daemon_blockmissing.log" 2>&1 &
wait $!
echo "$(date) submit-blockmiss finished (ALL BLOCKMISSING SUBMITTED)"
