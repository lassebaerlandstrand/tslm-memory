#!/usr/bin/env bash
# The whole proof of concept after scripts/select_queries.py and scripts/fetch.py, for tmux: tee
# to runs/run_all.log. Steps whose outputs exist are skipped (stage 1 then only re-evaluates its
# saved adapter). probe.py stops if gates_confirm.json is not committed.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $* $(date +%H:%M)"; uv run python "$@"; }

[ -f data/emb.npy ] || run scripts/embed.py
[ -f runs/encoder_check.json ] || run scripts/encoder_check.py
for seed in 0 1 2; do
    run scripts/stage1.py "$seed"
    # The probe does not start unless the 16 tokens carry the recording.
    uv run python -c "import json, sys; sys.exit(not json.load(open('runs/stage1_s$seed.json'))['pass'])"
    [ -f "runs/eval/confirm/probe-S-dates_s$seed.jsonl" ] || run scripts/probe.py "$seed"
done
run scripts/confirm_results.py
run scripts/make_results.py
run scripts/make_figures.py
