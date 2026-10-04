"""12-lead MIMIC-IV-ECG records at 500 Hz in PTB-XL lead order."""

from pathlib import Path

import numpy as np
import wfdb

DATA = Path(__file__).resolve().parents[2] / "data"
PTBXL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def read500(path: str) -> np.ndarray:
    """[5000, 12] in mV. MIMIC stores aVF before aVL, so leads are picked by name."""
    rec = wfdb.rdrecord(str(DATA / "mimic" / path))
    return rec.p_signal[:5000, [rec.sig_name.index(n) for n in PTBXL_LEADS]].astype(np.float32)


def unusable(sig: np.ndarray) -> bool:
    """A lead missing throughout, or every lead flat. Short dropouts are set to 0 elsewhere."""
    return bool(np.isnan(sig).all(axis=0).any() or (np.nanstd(sig, axis=0) < 1e-3).all())
