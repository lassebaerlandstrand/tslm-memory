#!/bin/bash
# One screen of status: every lane's last marker and whether it is alive, the GPU, the jobs.
cd "$(dirname "$0")/.."
for f in runs/queue/*.log; do
  [[ -e $f && $f != *.full.log ]] || continue
  lane=$(basename "$f" .log) pid=$(cat "runs/queue/$lane.pid" 2>/dev/null)
  state=ended
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && state="running (pid $pid)"
  echo "$lane: $state | $(grep '^==' "$f" | tail -1 | cut -c1-140)"
done
echo "gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
pgrep -af "python -m tslm_memory" | grep -v pgrep | cut -c1-140
