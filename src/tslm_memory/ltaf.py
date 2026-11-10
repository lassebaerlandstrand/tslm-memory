"""TS-Haystack on LTAF ECG (arXiv 2602.14200, github.com/AI-X-Labs/TS-Haystack): its questions,
the PhysioNet Long-Term AF Database records they point into, and the event list of each window.

Questions are nz00shuuuu/ltaf-haystack-fixed (rhythms), windows of one record at 128 Hz with
times given from the window's start. Records are split 67 / 8 / 9 by the benchmark."""

import glob
import json
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path("data/tshaystack")
HAYSTACK = ROOT / "ltaf_haystack"
RECORDS = ROOT / "ltafdb/files"
HZ = 128
TASKS = [
    "existence",
    "localization",
    "counting",
    "ordering",
    "state_query",
    "antecedent",
    "comparison",
    "multi_hop",
    "anomaly_detection",
    "anomaly_localization",
]
RHYTHMS = ["NSR", "AFIB", "SBR", "AB", "SVTA", "B", "VT", "T", "IVR"]


def questions(split: str, lengths: list[int]) -> pd.DataFrame:
    """All valid questions of a split ("train", "validation", "test") at the given lengths, in
    seconds."""
    out = []
    for s in lengths:
        for p in sorted(glob.glob(str(HAYSTACK / f"rhythms/tasks/{s}_0s/*/{split}/data.parquet"))):
            t = pq.read_table(p).to_pandas()
            if len(t):
                out.append(t.assign(length=s))
    q = pd.concat(out, ignore_index=True)
    q = q[q.is_valid].reset_index(drop=True)
    q["start"] = q.window_start_ms * HZ // 1000  # first sample of the window in the record
    return q


@cache
def gains(record: str) -> np.ndarray:
    lines = open(RECORDS / f"{record}.hea").read().splitlines()
    return np.array([float(line.split()[2].split("/")[0]) for line in lines[1:3]], np.float32)


@cache
def signal(record: str) -> np.memmap:
    """[samples, 2] int16, the record as stored (format 16, baseline 0); divide by gains for mV."""
    return np.memmap(RECORDS / f"{record}.dat", np.int16, "r").reshape(-1, 2)


@cache
def rhythm_timeline(record: str) -> pd.DataFrame:
    return pq.read_table(HAYSTACK / f"timelines/{record}.parquet").to_pandas()


@cache
def beat_timeline(record: str) -> pd.DataFrame:
    return pq.read_table(HAYSTACK / f"beat_timelines/{record}.parquet").to_pandas()


def events(record: str, start: int, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The window's annotated events, in samples from its start: rhythm bouts clipped to it and
    beats that are not N. Each carries its ordinal among events of its kind in the window."""
    t = rhythm_timeline(record)
    t = t[(t.end_sample > start) & (t.start_sample < start + n)]
    bouts = pd.DataFrame(
        {
            "label": t.activity.to_numpy(),
            "start": np.maximum(t.start_sample.to_numpy() - start, 0),
            "end": np.minimum(t.end_sample.to_numpy() - start, n),
        }
    )
    bouts["ordinal"] = bouts.groupby("label").cumcount() + 1
    b = beat_timeline(record)
    lo, hi = np.searchsorted(b["sample"].to_numpy(), [start, start + n])
    b = b.iloc[lo:hi]
    b = b[b.symbol != "N"]
    beats = pd.DataFrame({"label": b.symbol.to_numpy(), "start": b["sample"].to_numpy() - start})
    beats["ordinal"] = beats.groupby("label").cumcount() + 1
    return bouts, beats


def clock(samples: float) -> str:
    """Samples from the window start as the benchmark's HH:MM:SS.mmm."""
    ms = round(samples / HZ * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def meta(q: pd.Series) -> dict:
    return json.loads(q.metadata)
