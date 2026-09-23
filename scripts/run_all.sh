#!/usr/bin/env bash
# The whole proof of concept after scripts/fetch.py, for tmux: tee to runs/run_all.log.
# Stops before scoring the test set if gates.json is not committed.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $* $(date +%H:%M)"; uv run python "$@"; }

run scripts/embed.py
run scripts/encoder_check.py
run scripts/baselines.py
if ! git ls-files --error-unmatch gates.json >/dev/null 2>&1; then
    run scripts/gates.py
    git add gates.json && git commit -qm "Fix the gates before scoring the test set" -- gates.json
fi
git diff --quiet HEAD -- gates.json
for seed in 0 1 2; do
    run scripts/stage1.py "$seed"
    # Gate P1: stage 2 does not start unless the 16 tokens carry the recording.
    uv run python -c "import json, sys; sys.exit(not json.load(open('runs/stage1_s$seed.json'))['pass'])"
    run scripts/train.py none "$seed"
    run scripts/train.py ctx "$seed"
    run scripts/evaluate.py "$seed" val
done
for seed in 0 1 2; do
    run scripts/evaluate.py "$seed" test
done
run scripts/make_results.py
run scripts/make_figures.py
