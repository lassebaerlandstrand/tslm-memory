"""MIMIC-IV-ECG machine statements as canonical labels, next-ECG pairs, and the subject split.

Only canonical labels ever leave this module, so no raw machine text reaches a prompt or a log.
"""

import hashlib
import re

import numpy as np
import pandas as pd

from tslm_context.waves import DATA

# Qualified statements do not count as the finding ("incomplete RBBB" is not RBBB).
QUALIFIED = re.compile(
    r"incomplete|borderline|possible|probable|consider|cannot|no longer|replaced"
)

LABELS = {
    "sinus rhythm": r"sinus rhythm",
    "sinus tachycardia": r"sinus tachycardia",
    "sinus bradycardia": r"sinus bradycardia",
    "atrial fibrillation or flutter": r"atrial fibrillation|atrial flutter",
    "paced rhythm": r"pac(ed|ing|emaker)",
    "premature ventricular complexes": r"ventricular premature|premature ventricular|pvc",
    "premature atrial complexes": (
        r"atrial premature|premature atrial|pac\b|supraventricular extrasystole"
    ),
    "first degree av block": r"1st degree|first degree",
    "left bundle branch block": r"left bundle branch block|lbbb",
    "right bundle branch block": r"right bundle branch block|rbbb",
    "left anterior fascicular block": r"left anterior fascicular block|lafb",
    "left axis deviation": r"left axis deviation",
    "right axis deviation": r"right axis deviation",
    "low qrs voltage": r"low (qrs )?voltage",
    "left ventricular hypertrophy": r"left ventricular hypertrophy|lvh",
    "prolonged qt": r"prolonged qt|long qt|qt.{0,12}prolong",
    "st-t changes suggesting ischemia": r"isch",
    "st-t abnormality": r"st-t|st segment|t wave",
    "infarct pattern": r"infarct",
    "normal ecg": r"^normal ecg",
}

FINDINGS = [
    "atrial fibrillation or flutter",
    "sinus tachycardia",
    "prolonged qt",
    "st-t changes suggesting ischemia",
]


def labels_of(lines: list[str]) -> list[str]:
    found = []
    for line in lines:
        text = line.lower()
        if QUALIFIED.search(text):
            continue
        for name, pattern in LABELS.items():
            if name not in found and re.search(pattern, text):
                found.append(name)
    return found


MEASURES = ["hr", "pr", "qrs", "qt", "qtc", "p_axis", "qrs_axis", "t_axis"]


def measures(mm: pd.DataFrame) -> pd.DataFrame:
    """Intervals (ms), rate (bpm) and axes (degrees) from the machine's fiducial points. The
    machine writes 29999 and similar sentinels for missing points; values out of range are NaN."""

    def ok(x: pd.Series, lo: float, hi: float) -> pd.Series:
        return x.where((x >= lo) & (x <= hi))

    rr = ok(mm.rr_interval.astype(float), 200, 3000)
    out = pd.DataFrame(index=mm.index)
    out["hr"] = 60000 / rr
    out["pr"] = ok(mm.qrs_onset - mm.p_onset, 50, 500)
    out["qrs"] = ok(mm.qrs_end - mm.qrs_onset, 40, 300)
    out["qt"] = ok(mm.t_end - mm.qrs_onset, 200, 700)
    out["qtc"] = out.qt / np.sqrt(rr / 1000)  # Bazett
    for a in ("p_axis", "qrs_axis", "t_axis"):
        out[a] = ok(mm[a].astype(float), -180, 360)
    return out


def canonical_table() -> pd.DataFrame:
    """One row per ECG: subject, study, time, path, canonical labels and machine measures."""
    mm = pd.read_csv(DATA / "mimic" / "machine_measurements.csv", low_memory=False)
    rl = pd.read_csv(DATA / "mimic" / "record_list.csv", usecols=["study_id", "path"])
    reports = mm[[f"report_{i}" for i in range(18)]].astype("string").fillna("")
    mm["labels"] = [labels_of([x for x in row if x]) for row in reports.itertuples(index=False)]
    mm = pd.concat([mm, measures(mm)], axis=1)
    out = mm[["subject_id", "study_id", "ecg_time", "labels", *MEASURES]].merge(rl, on="study_id")
    out["ecg_time"] = pd.to_datetime(out.ecg_time)
    return out.sort_values(["subject_id", "ecg_time"]).reset_index(drop=True)


def next_pairs(ecgs: pd.DataFrame, min_h: float = 1.0, max_d: float = 365.0) -> pd.DataFrame:
    """For each ECG, the first later ECG of the same subject at least min_h hours and at most max_d
    days later. Times are only compared within one subject (MIMIC shifts dates per subject)."""
    rows = []
    for sid, g in ecgs.groupby("subject_id", sort=False):
        t = g.ecg_time.to_numpy()
        for i in range(len(g) - 1):
            gap = (t[i + 1 :] - t[i]) / np.timedelta64(1, "h")
            ok = np.flatnonzero(gap >= min_h)
            if len(ok) and gap[ok[0]] <= max_d * 24:
                j = i + 1 + ok[0]
                rows.append((sid, g.index[i], g.index[j], gap[ok[0]] / 24))
    return pd.DataFrame(rows, columns=["subject_id", "cur", "nxt", "gap_days"])


def earlier(ecgs: pd.DataFrame, rows: np.ndarray, n: int, min_h: float = 1.0) -> np.ndarray:
    """The n most recent ECGs of the same subject recorded at least min_h hours before each row,
    most recent first; -1 where there are fewer. ecgs must be sorted by subject and time."""
    code = pd.factorize(ecgs.subject_id)[0].astype(np.int64)
    minutes = (ecgs.ecg_time.to_numpy() - np.datetime64("2000-01-01")) // np.timedelta64(1, "m")
    key = code * 10**9 + minutes.astype(np.int64)
    assert (np.diff(key) >= 0).all()
    rows = np.asarray(rows)
    last = np.searchsorted(key, key[rows] - int(min_h * 60), side="right") - 1
    out = np.full((len(rows), n), -1, dtype=np.int64)
    for j in range(n):
        idx = last - j
        ok = (idx >= 0) & (code[np.maximum(idx, 0)] == code[rows])
        out[:, j] = np.where(ok, idx, -1)
    return out


def bucket(subject_id: int) -> int:
    """0 = test queries, 1 = validation, 2..9 = index pool. Stable across runs."""
    return int(hashlib.md5(str(subject_id).encode()).hexdigest(), 16) % 10
