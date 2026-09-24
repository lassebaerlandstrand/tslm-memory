"""MIMIC-IV-ECG machine statements as canonical labels, next-ECG pairs, own history, and the subject
split.

Only canonical labels ever leave this module, so no raw machine text reaches a prompt or a log.
"""

import hashlib
import re

import numpy as np
import pandas as pd

from tslm_memory.waves import DATA

# A qualifier makes a finding "possible" when it precedes the finding within the same clause
# ("probable atrial fibrillation"), not when it belongs to another clause of the line
# ("sinus tachycardia with borderline first degree A-V block").
QUALIFIER = re.compile(r"incomplete|borderline|possible|probable|consider|cannot")
CLAUSE = re.compile(r" with |,|;| and | - ")

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

MEASURES = ["hr", "pr", "qrs", "qt", "qtc", "p_axis", "qrs_axis", "t_axis"]


def labels_of(lines: list[str]) -> tuple[list[str], list[str]]:
    """(definite labels, labels mentioned only with a qualifier), in LABELS order."""
    definite, possible = set(), set()
    for line in lines:
        text = line.lower()
        for name, pattern in LABELS.items():
            for m in re.finditer(pattern, text):
                starts = [c.end() for c in CLAUSE.finditer(text, 0, m.start())]
                clause = text[starts[-1] if starts else 0 : m.start()]
                (possible if QUALIFIER.search(clause) else definite).add(name)
    return (
        [n for n in LABELS if n in definite],
        [n for n in LABELS if n in possible and n not in definite],
    )


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
    """One row per ECG, sorted by subject and time, with a RangeIndex: subject, study, time, path,
    canonical labels, qualified-only labels and machine measures."""
    mm = pd.read_csv(DATA / "mimic" / "machine_measurements.csv", low_memory=False)
    rl = pd.read_csv(DATA / "mimic" / "record_list.csv", usecols=["study_id", "path"])
    reports = mm[[f"report_{i}" for i in range(18)]].astype("string").fillna("")
    found = [labels_of([x for x in row if x]) for row in reports.itertuples(index=False)]
    mm["labels"] = [d for d, _ in found]
    mm["possible"] = [p for _, p in found]
    mm = pd.concat([mm.drop(columns=["p_axis", "qrs_axis", "t_axis"]), measures(mm)], axis=1)
    cols = ["subject_id", "study_id", "ecg_time", "labels", "possible", *MEASURES]
    out = mm[cols].merge(rl, on="study_id")
    out["ecg_time"] = pd.to_datetime(out.ecg_time)
    return out.sort_values(["subject_id", "ecg_time"]).reset_index(drop=True)


def _key(ecgs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row subject code and a sortable (subject, minute) key. Rows must be sorted by subject
    and time under a RangeIndex, so row labels and positions agree."""
    assert ecgs.index.equals(pd.RangeIndex(len(ecgs)))
    code = pd.factorize(ecgs.subject_id)[0].astype(np.int64)
    minutes = (ecgs.ecg_time.to_numpy() - np.datetime64("2000-01-01")) // np.timedelta64(1, "m")
    key = code * 10**9 + minutes.astype(np.int64)
    assert (np.diff(key) >= 0).all()
    return code, key


def next_pairs(ecgs: pd.DataFrame, min_h: float = 1.0, max_d: float = 365.0) -> pd.DataFrame:
    """For each ECG, the first later ECG of the same subject at least min_h hours and at most max_d
    days later. Times are only compared within one subject (MIMIC shifts dates per subject)."""
    code, key = _key(ecgs)
    rows = np.arange(len(ecgs))
    nxt = np.searchsorted(key, key + int(min_h * 60), side="left")
    ok = nxt < len(ecgs)
    ok[ok] &= code[nxt[ok]] == code[rows[ok]]
    gap = np.full(len(ecgs), np.inf)
    gap[ok] = (key[nxt[ok]] - key[rows[ok]]) / (60 * 24)
    ok &= gap <= max_d
    return pd.DataFrame(
        {
            "subject_id": ecgs.subject_id.to_numpy()[ok],
            "cur": rows[ok],
            "nxt": nxt[ok],
            "gap_days": gap[ok],
        }
    )


def earlier(ecgs: pd.DataFrame, rows: np.ndarray, n: int, min_h: float = 1.0) -> np.ndarray:
    """Own history: up to n earlier ECGs of the same subject, most recent first, each at least
    min_h hours before the previous one (the first before the row itself). -1 where there are
    fewer."""
    code, key = _key(ecgs)
    rows = np.asarray(rows)
    out = np.full((len(rows), n), -1, dtype=np.int64)
    ref = rows.copy()
    alive = np.ones(len(rows), dtype=bool)
    for j in range(n):
        idx = np.searchsorted(key, key[ref] - int(min_h * 60), side="right") - 1
        alive &= (idx >= 0) & (code[np.maximum(idx, 0)] == code[rows])
        out[alive, j] = idx[alive]
        ref = np.where(alive, idx, ref)
    return out


def bucket(subject_id: int) -> int:
    """0 = test queries, 1 = validation, 2..9 = training and pool. Stable across runs."""
    assert isinstance(subject_id, int | np.integer), type(subject_id)
    return int(hashlib.md5(str(int(subject_id)).encode()).hexdigest(), 16) % 10
