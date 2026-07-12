#!/bin/bash -l
#SBATCH --job-name=submit-sk
#SBATCH --partition=academic
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/submit-sk-%j.out
#SBATCH --error=logs/submit-sk-%j.err

# Durable SLURM-hosted runner for the sklearn-models submitter. Self-chains on
# USR1 ~180s before walltime so submission survives the 72h limit.
set -u
ROOT=/work/scitas-share/FellayMultiOmic/code/full-test-pipeline
cd "$ROOT"

requeue() {
  echo "$(date) walltime approaching — resubmitting submit-sk to continue"
  sbatch "$ROOT/slurm_submit_daemon_sk.sh"
  exit 0
}
trap requeue USR1

echo "$(date) submit-sk SLURM job $SLURM_JOB_ID starting"
bash "$ROOT/submit_daemon_sk.sh" >> "$ROOT/submit_daemon_sk.log" 2>&1 &
wait $!
echo "$(date) submit-sk finished (ALL SK SUBMITTED)"
