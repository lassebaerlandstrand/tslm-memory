#!/usr/bin/env bash
# The retrieval step after run_all.sh (same stage-1 adapters and seeds), for tmux: tee to
# runs/run_retrieval.log. Stops before scoring the test set if gates_retrieval.json is not committed.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $* $(date +%H:%M)"; uv run python "$@"; }

git diff --quiet HEAD -- gates_retrieval.json
[ -f data/emb_next.npy ] || run scripts/build_pool.py embed
run scripts/baselines.py ret
for seed in 0 1 2; do
    run scripts/train.py ret "$seed"
    run scripts/evaluate.py "$seed" val ret
done
for seed in 0 1 2; do
    run scripts/evaluate.py "$seed" test ret
done
run scripts/make_results.py
run scripts/make_figures.py
