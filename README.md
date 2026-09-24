# tslm-memory

Memory for time-series language models (TSLMs). A TSLM answers questions about one recording at a
time. Here it also reads the same person's earlier recordings, each compressed into a few tokens
the same way as the current one.

The proof of concept uses ECG. On MIMIC-IV-ECG, OpenTSLM-SoftPrompt on Llama-3.2-1B (its released
ECG checkpoint, kept frozen) reads each ECG as 16 tokens: a frozen ECGFounder embedding, a small
trained adapter and OpenTSLM's projector. A logistic regression on the model's state predicts
whether the machine read of the patient's next ECG reports atrial fibrillation or flutter, sinus
tachycardia, prolonged QT, or ST-T changes suggesting ischemia. On 3,200 patients scored after the
comparison was committed (`gates_confirm.json`), AUROC within finding and current state:

| Input | AUROC |
|---|---|
| Current ECG only | 0.704 |
| + only the dates of up to three earlier ECGs | 0.727 |
| + the earlier ECGs themselves | 0.755 |
| + another patient's earlier ECGs at the same dates | 0.647 |

## Setup

Requires [uv](https://docs.astral.sh/uv/); it installs Python 3.14 and all dependencies. The base
model (Llama-3.2-1B) is gated on Hugging Face: accept its licence and put `HF_TOKEN` in `.env`.

```bash
uv sync
uv run pytest
uv run ruff format . && uv run ruff check .
```

## Pipeline

`scripts/select_queries.py` chooses the queries and `scripts/fetch.py` downloads their records.
`scripts/run_all.sh` runs the rest: `embed.py` (ECGFounder embeddings), `encoder_check.py` (why
OpenTSLM's own ECG encoder is not used), `stage1.py` (the adapter, three seeds), `probe.py` (the
frozen reader read by a linear probe), `confirm_results.py`, and `make_results.py` and
`make_figures.py`, which write the proposal's numbers and figure.

## Data

MIMIC-IV-ECG 1.0 (ODbL, open access) and the ECGFounder weights (MIT). Nothing derived from them
is committed.
