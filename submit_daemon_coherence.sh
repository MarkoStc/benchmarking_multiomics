#!/bin/bash
# Throttled submitter for the README-coherence campaigns across all 7 models:
#   - blockmissing_hpgrid  (README 8.2 random block missingness, full HP grid)
#   - featomic_defaults    (README 9.3 one-omic-at-a-time feature sweep, default HP)
# Supersedes submit_daemon_blockmissing.sh: it resumes block-missing progress by
# seeding from that daemon's state file, then interleaves the per-omic files. Keeps
# the per-user queue under the QOS cap. Each model's slurm_*_array.sh handles env.
# Resumable via submit_daemon_coherence.state.
set -u

ROOT=/work/scitas-share/FellayMultiOmic/code/full-test-pipeline
STATE="$ROOT/submit_daemon_coherence.state"
BM_STATE="$ROOT/submit_daemon_blockmissing.state"
CAP=1801
MARGIN=40
MIN_SLICE=10
POLL=90

MODELS=(logreg_early svm_early pca_logreg logreg_late pnet mofa integrao)
AXES=(blockmissing_hpgrid featomic_defaults)
declare -A SLURM=(
  [logreg_early]=slurm_logreg_early_array.sh
  [svm_early]=slurm_svm_early_array.sh
  [pca_logreg]=slurm_pca_logreg_array.sh
  [logreg_late]=slurm_logreg_late_array.sh
  [pnet]=slurm_pnet_array.sh
  [mofa]=slurm_mofa_array.sh
  [integrao]=slurm_integrao_array.sh
)

# seed resume-points: our own state wins; else fall back to the block-missing state
declare -A DONE_NEXT=()
for src in "$BM_STATE" "$STATE"; do
  [ -f "$src" ] || continue
  while IFS='|' read -r m f nx; do
    [ -n "$m" ] && DONE_NEXT["$m|$f"]="$nx"
  done < "$src"
done

WL=()
for model in "${MODELS[@]}"; do
  for ds in TCGA-BRCA TCGA-LGG TCGA-KIPAN; do
    for axis in "${AXES[@]}"; do
      f="$ROOT/$model/experiments_${ds}_${axis}.txt"
      [ -s "$f" ] || continue
      n=$(wc -l < "$f")
      start="${DONE_NEXT[$model|$f]:-1}"
      WL+=("$model|${SLURM[$model]}|$f|$n|$start")
    done
  done
done

persist_state() {
  local e pm ps pf pn pnext
  : > "$STATE"
  for e in "${WL[@]}"; do
    IFS='|' read -r pm ps pf pn pnext <<< "$e"
    echo "$pm|$pf|$pnext" >> "$STATE"
  done
}

echo "$(date) START coherence daemon, ${#WL[@]} files, total $(printf '%s\n' "${WL[@]}" | awk -F'|' '{s+=$4} END{print s}') jobs (remaining below)"

submitted_total=0
while true; do
  cur=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)
  avail=$(( CAP - cur - MARGIN ))
  if [ "$avail" -lt "$MIN_SLICE" ]; then sleep "$POLL"; continue; fi

  progressed=0
  for i in "${!WL[@]}"; do
    IFS='|' read -r model slurm file n next <<< "${WL[$i]}"
    [ "$next" -gt "$n" ] && continue
    cur=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)
    avail=$(( CAP - cur - MARGIN ))
    [ "$avail" -lt "$MIN_SLICE" ] && break
    end=$(( next + avail - 1 ))
    [ "$end" -gt "$n" ] && end="$n"
    cd "$ROOT/$model" || continue
    jid=$(sbatch --array="${next}-${end}" "$slurm" "$file" 2>/dev/null | awk '{print $NF}')
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
      cnt=$(( end - next + 1 ))
      submitted_total=$(( submitted_total + cnt ))
      WL[$i]="$model|$slurm|$file|$n|$(( end + 1 ))"
      persist_state
      echo "$(date) submit $model $(basename "$file") [${next}-${end}] ($cnt) jid=$jid | cumulative=$submitted_total"
      progressed=1
    fi
  done

  remaining=0
  for e in "${WL[@]}"; do
    IFS='|' read -r _ _ _ n next <<< "$e"
    [ "$next" -le "$n" ] && remaining=$(( remaining + n - next + 1 ))
  done
  if [ "$remaining" -eq 0 ]; then
    echo "$(date) ALL COHERENCE CAMPAIGNS SUBMITTED. total=$submitted_total"
    break
  fi
  [ "$progressed" -eq 0 ] && sleep "$POLL"
done
