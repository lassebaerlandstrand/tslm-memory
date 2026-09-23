"""Gate G0, metadata only: does a patient's own history predict the finding on the next ECG
beyond the current ECG's machine read and measures? No waveforms and no language model.

Statistician = gradient-boosted trees per finding on (a) the current ECG's labels, measures and the
gap to the next ECG, and (b) the same plus the up to three most recent earlier ECGs. Scored as mean
AUROC within strata (finding x current state) on bucket-0 subjects, cell-balanced as the PoC test
set will be. Prints aggregates only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from tslm_context.grade import auroc_within, paired_bootstrap
from tslm_context.mimic import (
    FINDINGS,
    LABELS,
    MEASURES,
    bucket,
    canonical_table,
    earlier,
    next_pairs,
)

N_PRIOR, N_CELL, N_TRAIN = 3, 150, 60000


def features(ecgs: pd.DataFrame, pairs: pd.DataFrame, history: bool) -> np.ndarray:
    lab = np.array([[name in x for name in LABELS] for x in ecgs.labels], dtype=np.float32)
    num = ecgs[MEASURES].to_numpy(np.float32)
    cur = pairs.cur.to_numpy()
    cols = [lab[cur], num[cur], np.log1p(pairs.gap_days.to_numpy())[:, None]]
    if history:
        prev = np.stack([pairs[f"prior{j}"].to_numpy() for j in range(N_PRIOR)], axis=1)
        have = prev >= 0
        p0 = prev[:, 0]
        cols += [lab[p0], num[p0], np.log1p(pairs.since_prior.to_numpy())[:, None]]
        frac = sum(lab[np.where(have[:, j], prev[:, j], 0)] * have[:, [j]] for j in range(N_PRIOR))
        cols += [frac / have.sum(axis=1, keepdims=True), have.sum(axis=1, keepdims=True)]
    return np.concatenate(cols, axis=1)


def has(ecgs: pd.DataFrame, f: str, rows: np.ndarray) -> np.ndarray:
    return np.array([f in x for x in ecgs.labels.to_numpy()[rows]])


def main() -> None:
    ecgs = canonical_table()
    pairs = next_pairs(ecgs)
    prev = earlier(ecgs, pairs.cur.to_numpy(), N_PRIOR)
    for j in range(N_PRIOR):
        pairs[f"prior{j}"] = prev[:, j]
    pairs = pairs[pairs.prior1 >= 0].reset_index(drop=True)
    t = ecgs.ecg_time.to_numpy()
    assert (t[pairs.prior0] < t[pairs.cur]).all() and (t[pairs.prior1] <= t[pairs.prior0]).all()
    pairs["since_prior"] = (t[pairs.cur] - t[pairs.prior0]) / np.timedelta64(1, "D")
    pairs["bucket"] = pairs.subject_id.map(bucket)
    print(f"pairs with >=2 earlier ECGs {len(pairs)}, subjects {pairs.subject_id.nunique()}")

    one = pairs.sample(frac=1, random_state=0).drop_duplicates("subject_id")
    train = one[one.bucket >= 2].head(N_TRAIN)
    # Test cells draw from every bucket-0 pair (one per subject within a cell), as the PoC will.
    test_pool = pairs[pairs.bucket == 0].sample(frac=1, random_state=1)
    x_tr = {h: features(ecgs, train, h) for h in (False, True)}

    res, rows = {}, []
    for f in FINDINGS:
        # History carries information if the finding on an earlier ECG changes the transition rate.
        a, b = has(ecgs, f, one.cur), has(ecgs, f, one.nxt)
        prior_has = np.array(
            [
                any(f in ecgs.labels.iat[p] for p in r if p >= 0)
                for r in one[[f"prior{j}" for j in range(N_PRIOR)]].to_numpy()
            ]
        )
        for s in (False, True):
            m = a == s
            res[f"{f}|now={int(s)}|P(next|prior has)"] = float(b[m & prior_has].mean())
            res[f"{f}|now={int(s)}|P(next|prior not)"] = float(b[m & ~prior_has].mean())
        ya = has(ecgs, f, train.nxt)
        cells = []
        tp_now, tp_next = has(ecgs, f, test_pool.cur), has(ecgs, f, test_pool.nxt)
        for s in (False, True):
            for y in (False, True):
                cell = test_pool[(tp_now == s) & (tp_next == y)].drop_duplicates("subject_id")
                cells.append(cell.head(N_CELL))
        te = pd.concat(cells)
        for h in (False, True):
            clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=0)
            clf.fit(x_tr[h], ya)
            p = clf.predict_proba(features(ecgs, te, h))[:, 1]
            for i, (sid, y, s, pp) in enumerate(
                zip(te.subject_id, has(ecgs, f, te.nxt), has(ecgs, f, te.cur), p, strict=True)
            ):
                rows.append((f, h, i, sid, int(y), f"{f}|{int(s)}", pp))
    df = pd.DataFrame(rows, columns=["finding", "history", "i", "subject", "y", "stratum", "p"])
    a, b = df[~df.history].reset_index(drop=True), df[df.history].reset_index(drop=True)
    assert (a.subject.to_numpy() == b.subject.to_numpy()).all()
    y, st, subj = a.y.to_numpy(), a.stratum.to_numpy(), a.subject.to_numpy()
    for name, d in (("current only", a), ("with history", b)):
        res[f"auroc {name}"] = auroc_within(y, d.p.to_numpy(), st)
        for f in FINDINGS:
            m = a.finding.to_numpy() == f
            res[f"auroc {name} | {f}"] = auroc_within(y[m], d.p.to_numpy()[m], st[m])

    def diff(i: np.ndarray) -> float:
        return auroc_within(y[i], b.p.to_numpy()[i], st[i]) - auroc_within(
            y[i], a.p.to_numpy()[i], st[i]
        )

    res["history minus current"] = diff(np.arange(len(a)))
    res["history minus current CI"] = paired_bootstrap(diff, subj, n=1000)
    res["n test"] = len(a)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/g0.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v if not isinstance(v, float) else round(v, 3))


if __name__ == "__main__":
    main()
