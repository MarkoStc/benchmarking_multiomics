#!/bin/bash -l
#SBATCH --job-name=submit-daemon
#SBATCH --partition=academic
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/submit-daemon-%j.out
#SBATCH --error=logs/submit-daemon-%j.err

# Runs the throttled submitter (submit_daemon.sh) as a SLURM job so it cannot
# be reaped by the login node. The daemon is resumable via submit_daemon.state.
# If walltime is reached before ALL SUBMITTED, this script re-queues itself
# (trap on USR1, fired 180s before the limit) so submission continues seamlessly.
set -u

ROOT=/work/scitas-share/FellayMultiOmic/code/full-test-pipeline
cd "$ROOT"

requeue() {
  echo "$(date) walltime approaching — resubmitting submit-daemon to continue"
  sbatch "$ROOT/slurm_submit_daemon.sh"
  exit 0
}
trap requeue USR1

echo "$(date) submit-daemon SLURM job $SLURM_JOB_ID starting"
bash "$ROOT/submit_daemon.sh" >> "$ROOT/submit_daemon.log" 2>&1 &
DAEMON_PID=$!
wait "$DAEMON_PID"
echo "$(date) submit-daemon finished (daemon exited cleanly — ALL SUBMITTED)"
