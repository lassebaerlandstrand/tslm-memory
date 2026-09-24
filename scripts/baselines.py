"""No-model references for readout B: gradient-boosted trees per finding.

- T-current / T-history: the machine read (labels, qualified labels, measures) of the query ECG
  and the gap; plus each earlier ECG's machine read and offset.
- S-current / S-history: the same from the ECGFounder embedding (32 PCA components) instead of
  the machine read. The matched no-text reference.
- T-lookup: T-current plus the share of earlier ECGs carrying each label.
- T-current-42k / T-history-42k: trained on one pair per subject from all of buckets 2-9, the
  ceiling and the metadata-only gate G0 (does own history add beyond the current machine read?).
Matched models train on exactly the reader's training queries of that finding. Writes one JSON line
per query to runs/baselines/{role}/{name}.jsonl and prints mean AUROC within strata.
With `ret`: T-history+votes and S-history+votes add the share of the 4 most similar pool entries
whose next ECG reports the finding (the retrieval reference, gates_retrieval.json).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier

from tslm_context.grade import auroc_within
from tslm_context.mimic import FINDINGS, LABELS, MEASURES, bucket, earlier, next_pairs
from tslm_context.retrieval import WithNeighbours
from tslm_context.samples import N_HIST, Queries
from tslm_context.waves import DATA

NAMES = ["T-current", "T-history", "T-lookup", "S-current", "S-history"]


class Features:
    def __init__(self, ecgs: pd.DataFrame, emb: np.ndarray, row: pd.Series, pca: PCA):
        self.lab = {
            s: np.array([n in a for n in LABELS] + [n in b for n in LABELS], dtype=np.float32)
            for s, a, b in zip(ecgs.index, ecgs.labels, ecgs.possible, strict=True)
        }
        self.num = ecgs[MEASURES]
        self.emb, self.row, self.pca = emb, row, pca

    def text(self, s: int) -> np.ndarray:
        if s < 0:
            return np.full(2 * len(LABELS) + len(MEASURES), np.nan, dtype=np.float32)
        return np.concatenate([self.lab[s], self.num.loc[s].to_numpy(np.float32)])

    def signal(self, s: int) -> np.ndarray:
        if s < 0:
            return np.full(self.pca.n_components_, np.nan, dtype=np.float32)
        return self.pca.transform(self.emb[[self.row[s]]])[0].astype(np.float32)

    def __call__(self, name: str, cur: int, hist: list, days: list, gap: float) -> np.ndarray:
        kind = self.text if name.startswith("T") else self.signal
        x = [kind(cur), [np.log1p(gap)]]
        if "history" in name:
            for j in range(N_HIST):
                x += [kind(hist[j] if j < len(hist) else -1)]
                x += [[np.log1p(days[j]) if j < len(days) else np.nan]]
            x += [[len(hist)]]
        if name == "T-lookup":
            share = np.mean([self.lab[s][: len(LABELS)] for s in hist], axis=0)
            x += [share, [len(hist)]]
        return np.concatenate([np.asarray(v, dtype=np.float32) for v in x])


def matrix(feat: Features, name: str, q: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            feat(name, int(r.cur_study), list(r.earlier), list(r.days), float(r.gap_days))
            for r in q.itertuples()
        ]
    )


def pool_pairs(ecgs_by_row: pd.DataFrame) -> pd.DataFrame:
    """One pair per subject from buckets 2-9 with >= 2 earlier ECGs (the 42k ceiling set)."""
    pairs = next_pairs(ecgs_by_row)
    h = earlier(ecgs_by_row, pairs.cur.to_numpy(), N_HIST)
    keep = h[:, 1] >= 0
    pairs, h = pairs[keep].reset_index(drop=True), h[keep]
    pairs = pairs[[bucket(s) >= 2 for s in pairs.subject_id.to_numpy()]]
    pairs = pairs.sample(frac=1, random_state=0).drop_duplicates("subject_id")
    h = h[pairs.index.to_numpy()]
    study = ecgs_by_row.study_id.to_numpy()
    t = ecgs_by_row.ecg_time.to_numpy()
    out = pd.DataFrame(
        {
            "cur_study": study[pairs.cur],
            "next_study": study[pairs.nxt],
            "gap_days": pairs.gap_days.to_numpy(),
        }
    )
    out["earlier"] = [[int(study[x]) for x in row if x >= 0] for row in h]
    out["days"] = [
        [float((t[c] - t[x]) / np.timedelta64(1, "h") / 24) for x in row if x >= 0]
        for c, row in zip(pairs.cur, h, strict=True)
    ]
    return out


def main() -> None:
    tr = Queries("train")
    ecgs = tr.ecgs
    emb = np.load(DATA / "emb.npy")
    train_rows = tr.row[tr.q.cur_study].to_numpy()
    pca = PCA(32, random_state=0).fit(emb[train_rows])
    feat = Features(ecgs, emb, tr.row, pca)
    ceiling = pool_pairs(ecgs.reset_index())
    x_ceiling = {n: matrix(feat, n, ceiling) for n in ("T-current-42k", "T-history-42k")}
    print(f"matched training queries {len(tr)}, ceiling training pairs {len(ceiling)}")
    for role in ("val", "test", "confirm"):
        ev = Queries(role)
        out = Path("runs/baselines") / role
        out.mkdir(parents=True, exist_ok=True)
        for name in [*NAMES, "T-current-42k", "T-history-42k"]:
            p = np.zeros(len(ev))
            for f in FINDINGS:
                if name.endswith("42k"):
                    x_tr = x_ceiling[name]
                    y_tr = np.array([f in ecgs.labels[s] for s in ceiling.next_study])
                else:
                    sub = tr.q[tr.q.finding == f]
                    x_tr = matrix(feat, name, sub)
                    y_tr = np.array([c[1] == "1" for c in sub.cell])
                clf = HistGradientBoostingClassifier(
                    max_iter=300, early_stopping=True, random_state=0
                )
                clf.fit(x_tr, y_tr)
                m = (ev.q.finding == f).to_numpy()
                p[m] = clf.predict_proba(matrix(feat, name, ev.q[m]))[:, 1]
            y = np.array([ev.label(i) for i in range(len(ev))])
            strata = np.array([ev.stratum(i) for i in range(len(ev))])
            recs = [
                {"i": i, "stratum": strata[i], "y": int(y[i]), "p_yes": float(p[i])}
                for i in range(len(ev))
            ]
            (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            print(
                f"{role} {name}: AUROC within strata {auroc_within(y, p, strata):.3f}", flush=True
            )


def votes(q: WithNeighbours, k: int = 4) -> np.ndarray:
    """Share of the k most similar pool entries whose next ECG reports the finding asked about."""
    return np.array(
        [
            np.mean([f in q.ecgs.loc[int(q.pool_next[s]), "labels"] for s in q.nb[i][:k]])
            for i, f in enumerate(q.q.finding)
        ]
    )


def main_ret() -> None:
    tr = WithNeighbours("train")
    emb = np.load(DATA / "emb.npy")
    pca = PCA(32, random_state=0).fit(emb[tr.row[tr.q.cur_study].to_numpy()])
    feat = Features(tr.ecgs, emb, tr.row, pca)
    v_tr = votes(tr)
    for role in ("val", "test"):
        ev = WithNeighbours(role)
        v_ev = votes(ev)
        out = Path("runs/baselines") / role
        for base in ("T-history", "S-history"):
            p = np.zeros(len(ev))
            for f in FINDINGS:
                m_tr = (tr.q.finding == f).to_numpy()
                x_tr = np.c_[matrix(feat, base, tr.q[m_tr]), v_tr[m_tr]]
                y_tr = np.array([c[1] == "1" for c in tr.q.cell[m_tr]])
                clf = HistGradientBoostingClassifier(
                    max_iter=300, early_stopping=True, random_state=0
                ).fit(x_tr, y_tr)
                m = (ev.q.finding == f).to_numpy()
                p[m] = clf.predict_proba(np.c_[matrix(feat, base, ev.q[m]), v_ev[m]])[:, 1]
            y = np.array([ev.label(i) for i in range(len(ev))])
            strata = np.array([ev.stratum(i) for i in range(len(ev))])
            recs = [
                {"i": i, "stratum": strata[i], "y": int(y[i]), "p_yes": float(p[i])}
                for i in range(len(ev))
            ]
            (out / f"{base}+votes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
            print(f"{role} {base}+votes: AUROC within strata {auroc_within(y, p, strata):.3f}")


if __name__ == "__main__":
    main_ret() if sys.argv[1:] == ["ret"] else main()
