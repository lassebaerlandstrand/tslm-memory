#!/bin/bash
# Block while a lane's process is alive, then print the status. It watches the process, never
# an output file, so a crash cannot leave it waiting forever.
# usage: scripts/lane_wait.sh <lane>
cd "$(dirname "$0")/.."
pid=$(cat "runs/queue/$1.pid")
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
tail -5 "runs/queue/$1.log"
scripts/lane_status.sh
