"""Readout-B queries: which ECGs are asked about, with which history, in which role.

A query is an ECG with at least 2 earlier ECGs of the same subject (own history, up to 3, each at
least 1 h before the previous) and a next ECG (1 h to 365 d later). For each finding F and cell (F
on this ECG, F on the next), queries are drawn per role: test from bucket 0, validation from
bucket 1, training from buckets 2-9, plus a reserve per cell for records that turn out flat or
missing. Cells are filled rarest first and every subject is used once across all cells and roles.
Pairs where F is mentioned only with a qualifier on the current or the next ECG stay out of F's
cells. Within a cell, subjects are drawn uniformly, then one of their eligible pairs.
"""

import math

import numpy as np
import pandas as pd

from tslm_context.mimic import FINDINGS, bucket, earlier, next_pairs

N_HIST = 3
BUCKETS = {"test": {0}, "val": {1}, "train": set(range(2, 10))}


def select(
    ecgs: pd.DataFrame, per_cell: dict[str, int], reserve: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pairs = next_pairs(ecgs)
    hist = earlier(ecgs, pairs.cur.to_numpy(), N_HIST)
    keep = hist[:, 1] >= 0
    pairs, hist = pairs[keep].reset_index(drop=True), hist[keep]
    pairs["bucket"] = [bucket(s) for s in pairs.subject_id.to_numpy()]

    labels, possible = ecgs.labels.to_numpy(), ecgs.possible.to_numpy()
    cur, nxt = pairs.cur.to_numpy(), pairs.nxt.to_numpy()
    cells = []
    for f in FINDINGS:
        now = np.array([f in x for x in labels[cur]])
        then = np.array([f in x for x in labels[nxt]])
        vague = np.array([f in x for x in possible[cur]]) | np.array(
            [f in x for x in possible[nxt]]
        )
        for role, bk in BUCKETS.items():
            in_role = pairs.bucket.isin(bk).to_numpy() & ~vague
            for a in (False, True):
                for b in (False, True):
                    cells.append((role, f, f"{int(a)}{int(b)}", in_role & (now == a) & (then == b)))

    used: set[int] = set()
    rows = []
    for role, f, cell, m in sorted(cells, key=lambda c: pairs.subject_id[c[3]].nunique()):
        n = math.ceil(per_cell[role] * (1 + reserve))
        cand = pairs[m & ~pairs.subject_id.isin(used).to_numpy()]
        options = cand.groupby("subject_id").groups
        subjects = np.array(sorted(options))
        assert len(subjects) >= n, (role, f, cell, len(subjects), n)
        pick = rng.choice(subjects, n, replace=False)
        for rank, s in enumerate(pick):
            rows.append((role, f, cell, rank, int(rng.choice(options[s]))))
        used.update(int(s) for s in pick)

    q = pd.DataFrame(rows, columns=["role", "finding", "cell", "rank", "pair"])
    p = pairs.loc[q.pair]
    study, t = ecgs.study_id.to_numpy(), ecgs.ecg_time.to_numpy()
    q["subject_id"] = p.subject_id.to_numpy()
    q["cur_study"] = study[p.cur]
    q["next_study"] = study[p.nxt]
    q["gap_days"] = p.gap_days.to_numpy()
    h = hist[q.pair.to_numpy()]
    for j in range(N_HIST):
        ok = h[:, j] >= 0
        q[f"hist_study_{j}"] = np.where(ok, study[h[:, j]], -1)
        q[f"hist_days_{j}"] = np.where(ok, (t[p.cur] - t[h[:, j]]) / np.timedelta64(1, "D"), np.nan)
    q["n_hist"] = (h >= 0).sum(axis=1)
    q = q.drop(columns="pair")
    q.attrs["n_pairs"] = len(pairs)
    return q


def to_fetch(q: pd.DataFrame) -> np.ndarray:
    """Study ids whose waveforms are needed (query and history), test first. Next ECGs are never
    fetched."""
    out = []
    for role in ("test", "val", "train"):
        r = q[q.role == role]
        s = np.concatenate([r.cur_study] + [r[f"hist_study_{j}"] for j in range(N_HIST)])
        out.append(s[s >= 0])
    return np.concatenate(out)
