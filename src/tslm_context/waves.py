"""12-lead waveforms at 100 Hz in one memory-mapped array per source, as OpenTSLM reads them."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import wfdb

DATA = Path(__file__).resolve().parents[2] / "data"
PTBXL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def read_100hz(path: str, leads: list[str] | None = None) -> np.ndarray:
    # OpenTSLM decimates the 500 Hz record by taking every fifth sample.
    rec = wfdb.rdrecord(path)
    sig = rec.p_signal
    if leads is not None:
        sig = sig[:, [rec.sig_name.index(name) for name in leads]]
    return sig[::5][:1000].astype(np.float32)


def build(paths: list[str], out: Path, leads: list[str] | None = None) -> np.ndarray:
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(len(paths), 1000, 12))
    with ProcessPoolExecutor(16) as ex:
        for i, sig in enumerate(ex.map(read_100hz, paths, [leads] * len(paths), chunksize=64)):
            arr[i] = sig
    arr.flush()
    return arr


def load(name: str) -> np.ndarray:
    return np.load(DATA / f"{name}.npy", mmap_mode="r")
