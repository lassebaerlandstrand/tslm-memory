#!/usr/bin/env bash
# The whole proof of concept after scripts/fetch.py, for tmux: tee to runs/run_all.log.
# Stops before scoring the test set if gates.json is not committed. Resumes: steps whose outputs
# exist are skipped (stage 1 then only re-evaluates its saved adapter).
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $* $(date +%H:%M)"; uv run python "$@"; }

[ -f data/emb.npy ] || run scripts/embed.py
[ -f runs/encoder_check.json ] || run scripts/encoder_check.py
[ -f runs/baselines/test/T-history-42k.jsonl ] || run scripts/baselines.py
if ! git ls-files --error-unmatch gates.json >/dev/null 2>&1; then
    run scripts/gates.py
    git add gates.json && git commit -qm "Fix the gates before scoring the test set" -- gates.json
fi
git diff --quiet HEAD -- gates.json
for seed in 0 1 2; do
    run scripts/stage1.py "$seed"
    # Gate P1: stage 2 does not start unless the 16 tokens carry the recording.
    uv run python -c "import json, sys; sys.exit(not json.load(open('runs/stage1_s$seed.json'))['pass'])"
    [ -f "runs/train/none_s$seed/model.pt" ] || run scripts/train.py none "$seed"
    [ -f "runs/train/ctx_s$seed/model.pt" ] || run scripts/train.py ctx "$seed"
    run scripts/evaluate.py "$seed" val
done
for seed in 0 1 2; do
    run scripts/evaluate.py "$seed" test
done
run scripts/make_results.py
run scripts/make_figures.py
