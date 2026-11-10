#!/bin/bash
# Run a lane of jobs in order, detached from the shell that starts it, so the shell can close
# without stopping it. One "python args" per line of the jobs file, run as
# "uv run python <line>". The lane stops at the first failure and waits for GPU memory before
# each job. Markers "== start", "== FAILED", "== DONE" go to runs/queue/<lane>.log, every job's
# full output to runs/queue/<lane>.full.log, and the lane's PID to runs/queue/<lane>.pid.
# usage: scripts/lane.sh <lane> <jobs file> [MiB in use above which a job waits, default 4000]
set -u
cd "$(dirname "$0")/.."
lane=$1 jobs=$2 limit=${3:-4000}
mkdir -p runs/queue
if [ -z "${LANE_DETACHED:-}" ]; then
  LANE_DETACHED=1 setsid nohup "$0" "$@" > /dev/null 2>&1 < /dev/null &
  echo $! > "runs/queue/$lane.pid"
  echo "lane $lane started, pid $!"
  exit 0
fi
log=runs/queue/$lane.log full=runs/queue/$lane.full.log
echo "== $(date '+%d %b %H:%M') lane started, $(grep -cv -e '^#' -e '^$' "$jobs") jobs" >> "$log"
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
while read -r line; do
  [[ -z $line || $line == \#* ]] && continue
  while [ "$(gpu_used)" -gt "$limit" ]; do sleep 60; done
  echo "== $(date '+%d %b %H:%M') start $line" >> "$log"
  # stdin closed, or the job would read the rest of the jobs file.
  uv run python $line < /dev/null >> "$full" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    grep -v -i warn "$full" | tail -8 >> "$log"
    echo "== $(date '+%d %b %H:%M') FAILED (exit $rc) $line" >> "$log"
    exit 1
  fi
done < "$jobs"
echo "== $(date '+%d %b %H:%M') DONE" >> "$log"
