"""Readout-B samples: will the machine read of this patient's next ECG report finding F?

A sample is a list of segments, text or a record (a row of the ECGFounder embedding table that the
reader turns into 16 tokens). Two conditions:
- signal: every recording enters only as its 16 tokens; no machine read anywhere;
- text: every recording also carries its machine read (canonical labels and measures).
Own history is the query subject's earlier ECGs, most recent first. The swap control keeps the
query's own offsets and count and takes the recordings from a partner in the same stratum.
"""

import numpy as np
import pandas as pd

from tslm_context.mimic import MEASURES
from tslm_context.waves import DATA

N_HIST = 3
PER_CELL = {"test": 200, "val": 25, "train": 500}
NAMES = {"hr": "rate", "pr": "PR", "qrs": "QRS", "qt": "QT", "qtc": "QTc"}
UNITS = {"hr": "bpm", "pr": "ms", "qrs": "ms", "qt": "ms", "qtc": "ms"}


def when(days: float) -> str:
    return f"{days * 24:.0f} hours" if days < 1 else f"{days:.0f} days"


def machine_read(row: pd.Series) -> str:
    labels = list(row["labels"]) + [f"possible {x}" for x in row["possible"]]
    parts = [
        f"{NAMES[m]} n/a" if np.isnan(row[m]) else f"{NAMES[m]} {round(row[m])} {UNITS[m]}"
        for m in MEASURES[:5]
    ]
    axes = ", ".join(
        f"{a} {'n/a' if np.isnan(row[c]) else round(row[c])}"
        for a, c in (("P", "p_axis"), ("QRS", "qrs_axis"), ("T", "t_axis"))
    )
    statement = "; ".join(labels) if labels else "no statement"
    return f" Machine read: {statement}. Measures: {', '.join(parts)}, axes {axes}."


def swap_partners(strata: np.ndarray, n_hist: np.ndarray, seed: int = 0) -> np.ndarray:
    """For each query, another query of the same stratum (finding, current state) with the same
    number of history items, drawn without looking at the answer; -1 if it has no such peer."""
    rng = np.random.default_rng(seed)
    out = np.full(len(strata), -1)
    for s in np.unique(strata):
        for n in np.unique(n_hist[strata == s]):
            members = np.flatnonzero((strata == s) & (n_hist == n))
            if len(members) < 2:
                continue
            perm = rng.permutation(members)
            # A cyclic shift of a random permutation never maps a query to itself.
            out[perm] = np.roll(perm, 1)
    return out


class Queries:
    """Queries of one role with usable records only, `per_cell[role]` per (finding, cell). Reads
    data/ unless frames (queries, canonical table indexed by study_id, embedding row per study)
    are given."""

    def __init__(
        self,
        role: str,
        frames: tuple[pd.DataFrame, pd.DataFrame, pd.Series] | None = None,
        per_cell: dict[str, int] = PER_CELL,
    ):
        if frames is None:
            frames = (
                pd.read_parquet(DATA / "queries.parquet"),
                pd.read_parquet(DATA / "ecgs.parquet").set_index("study_id"),
                pd.read_parquet(DATA / "emb.parquet").set_index("study_id").row,
            )
        q, self.ecgs, idx = frames
        q = q[q.role == role]
        q = q[q.cur_study.map(idx).fillna(-1).to_numpy() >= 0].copy()
        # Unusable earlier records drop out of the history; the rest keep their order.
        hist, days = [], []
        for r in q.itertuples():
            pairs = [
                (int(getattr(r, f"hist_study_{j}")), float(getattr(r, f"hist_days_{j}")))
                for j in range(N_HIST)
            ]
            pairs = [(s, d) for s, d in pairs if s >= 0 and idx.get(s, -1) >= 0]
            hist.append([s for s, _ in pairs])
            days.append([d for _, d in pairs])
        q["earlier"], q["days"] = hist, days
        q = q[q.earlier.map(len) >= 1]
        q = q.sort_values("rank").groupby(["finding", "cell"]).head(per_cell[role])
        self.q = q.sort_values(["finding", "cell", "rank"]).reset_index(drop=True)
        self.row = idx
        self.role = role

    def __len__(self) -> int:
        return len(self.q)

    def label(self, i: int) -> int:
        return int(self.q.cell[i][1])

    def stratum(self, i: int) -> str:
        return f"{self.q.finding[i]}|{self.q.cell[i][0]}"

    def swap_partners(self, seed: int = 0) -> np.ndarray:
        strata = np.array([self.stratum(i) for i in range(len(self))])
        return swap_partners(strata, self.q.earlier.map(len).to_numpy(), seed)

    def _record(self, study: int, world: str) -> list:
        seg = [("ecg", int(self.row[study]))]
        if world == "text":
            seg.append(("text", machine_read(self.ecgs.loc[study])))
        return seg

    def sample(
        self,
        i: int,
        world: str,
        k: int,
        partner: int | None = None,
        oracle: bool = False,
        query_from: int | None = None,
    ) -> dict:
        """k earlier ECGs (own, or from `partner` with the query's own offsets); `oracle` adds the
        next ECG's machine read; `query_from` replaces the query recording (gate P2)."""
        r = self.q.iloc[i]
        cur = r.cur_study if query_from is None else self.q.cur_study[query_from]
        seg = [("text", "You are an expert cardiologist reading ECGs of one patient.\n\nThis ECG:")]
        seg += self._record(int(cur), world)
        k = min(k, len(r.earlier))
        if k:
            src = self.q.iloc[partner] if partner is not None else r
            assert len(src.earlier) >= k
            seg.append(("text", "\n\nEarlier ECGs of the same patient:"))
            for j in range(k):
                seg.append(("text", f"\n- {when(r.days[j])} before this one:"))
                seg += self._record(int(src.earlier[j]), world)
        if oracle:
            nxt = self.ecgs.loc[int(r.next_study)]
            seg.append(
                (
                    "text",
                    f"\n\nNext ECG of this patient, {when(r.gap_days)} later:{machine_read(nxt)}",
                )
            )
        seg.append(
            (
                "text",
                f"\n\nQuestion: This patient's next ECG was recorded {when(r.gap_days)} after this "
                f"one. Will its machine read report {r.finding}?\nAnswer yes or no.\nAnswer:",
            )
        )
        return {"segments": seg, "answer": " yes" if self.label(i) else " no"}
