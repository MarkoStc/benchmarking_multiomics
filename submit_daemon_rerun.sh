#!/bin/bash
# Throttled submitter for the RERUN of OOM-failed configs. Feeds rerun_*.txt
# using the 64G slurm_<model>_rerun.sh scripts, respecting the academic QOS cap.
# Resumable via submit_daemon_rerun.state. Same design as submit_daemon.sh.
set -u

ROOT=/work/scitas-share/FellayMultiOmic/code/full-test-pipeline
STATE="$ROOT/submit_daemon_rerun.state"
CAP=1801
MARGIN=30
MIN_SLICE=10
POLL=90

declare -A SLURM=( [integrao]=slurm_integrao_rerun.sh [pnet]=slurm_pnet_rerun.sh [mofa]=slurm_mofa_rerun.sh )

declare -A DONE_NEXT=()
if [ -f "$STATE" ]; then
  while IFS='|' read -r m f nx; do
    [ -n "$m" ] && DONE_NEXT["$m|$f"]="$nx"
  done < "$STATE"
fi

WL=()
for model in integrao pnet mofa; do
  for ds in TCGA-BRCA TCGA-LGG TCGA-KIPAN; do
    for axis in missing_hpgrid ratio_hpgrid nomics_defaults npatients_defaults; do
      f="$ROOT/$model/rerun_${ds}_${axis}.txt"
      [ -s "$f" ] || continue
      n=$(wc -l < "$f")
      key="$model|$f"
      start="${DONE_NEXT[$key]:-1}"
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

echo "$(date) START rerun daemon, ${#WL[@]} files, total $(printf '%s\n' "${WL[@]}" | awk -F'|' '{s+=$4} END{print s}') jobs"

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
    echo "$(date) ALL RERUN SUBMITTED. total=$submitted_total"
    break
  fi
  [ "$progressed" -eq 0 ] && sleep "$POLL"
done
