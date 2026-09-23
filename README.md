# tslm-context

A time-series language model that reads other recordings: the same person's earlier recordings
and similar people's recordings together with what followed them. Each context recording passes
through the model's own signal encoder and a small trained resampler into a few tokens. Proof of
concept on ECG with OpenTSLM (Stanford thesis proposal, January to June 2027).

## Setup

Requires [uv](https://docs.astral.sh/uv/); it installs Python 3.14 and all dependencies. The base
model (Llama-3.2-1B) is gated on Hugging Face: accept its licence and put `HF_TOKEN` in `.env`.

```bash
uv sync
uv run pytest
uv run ruff format . && uv run ruff check .
```

## Data

MIMIC-IV-ECG 1.0 (ODbL, open access) and the ECGFounder weights (MIT). Nothing derived from them
is committed.
