#!/bin/bash
# Train and score every system behind the report's tables, one job at a time, then build the
# tables and the figure under runs/report. The data must be prepared first, see the README.
# A run that already has its test results is skipped, so the script can be started again.
# usage: scripts/reproduce.sh [seeds, default "0 1 2"]
set -eu
cd "$(dirname "$0")/.."
seeds=${1:-"0 1 2"}
run() { uv run python -m "tslm_memory.$@"; }

# London trains in two phases. The first has 15000 steps over the last 200 days, without the
# month-change family, which needs a year. The second has 20000 steps over 830 days with every
# family and continues from the first's weights and optimizer state.
early="recall last_time count baseline control price_last month_total"
late="--window 830 --lr 1e-4 --gate-lr 1e-4 --lora-lr 1e-4 --steps 20000"
london() { # london <name> <seed> <arguments of both phases>
  local name=$1 seed=$2
  shift 2
  [ -e "runs/london/$name/test.npz" ] && return
  run train "$@" --window 200 --families $early --steps 15000 --seed "$seed" --name "${name}a"
  run train "$@" $late --init "${name}a" --seed "$seed" --name "$name"
  run evaluate "$name"
}
for s in $seeds; do
  london "memory_s$s" "$s" --mode full
  london "compression_s$s" "$s" --mode full --compress 64
  london "none_s$s" "$s" --mode none
done
london no_retrieval_s0 0 --mode full --retrieval-weight 0

# TS-Haystack, 8000 steps each. --test scores the test split at the end.
haystack() { # haystack <name> <arguments>
  local name=$1
  shift
  [ -e "runs/haystack/$name/test.npz" ] && return
  run haystack "$@" --test --name "$name"
}
for s in $seeds; do
  tiles="--bank tiles --perceive 1 --recall 8 --seed $s"
  events="--bank events --recall 4 --seed $s"
  haystack "tiles_s$s" $tiles
  haystack "tiles_compression_s$s" $tiles --compress 64
  haystack "events_s$s" $events
  haystack "events_compression_s$s" $events --compress 64
  haystack "none_s$s" --bank none --seed "$s"
done

run report
