import numpy as np
import pandas as pd
import pytest

from tslm_memory.samples import Queries, swap_partners
from tslm_memory.selection import select

PER_CELL = {"test": 6, "val": 1, "train": 3}


def test_swap_partner_is_blind_to_the_answer_and_matches_count():
    strata = np.array(["a"] * 300 + ["b"] * 300)
    n_hist = np.tile([2, 3], 300)
    y = np.r_[np.zeros(150), np.ones(150), np.zeros(150), np.ones(150)]  # ordered by answer
    p = swap_partners(strata, n_hist, seed=0)
    assert (p != np.arange(600)).all() and (p >= 0).all()
    assert (strata[p] == strata).all() and (n_hist[p] == n_hist).all()
    assert 0.4 <= (y[p] == y).mean() <= 0.6


def test_a_query_alone_with_its_history_count_borrows_a_richer_partner():
    strata = np.array(["a"] * 5)
    n_hist = np.array([1, 2, 2, 3, 3])
    p = swap_partners(strata, n_hist, seed=0)
    assert p[0] in (1, 2, 3, 4) and n_hist[p[0]] >= 1
    assert (p != np.arange(5)).all() and (n_hist[p] >= n_hist).all()


@pytest.fixture(scope="module")
def queries(ecgs):
    q = select(ecgs, PER_CELL, reserve=0.0)
    by_study = ecgs.set_index("study_id")
    # The next ECG has no embedding row, as it is never fetched.
    fetched = set(q.cur_study) | {s for j in range(3) for s in q[f"hist_study_{j}"] if s >= 0}
    row = pd.Series(np.arange(len(by_study)), index=by_study.index)
    row[~row.index.isin(list(fetched))] = -1
    return Queries("test", (q, by_study, row), PER_CELL), q


def text_of(sample: dict) -> str:
    return "".join(x for kind, x in sample["segments"] if kind == "text")


def rows_of(sample: dict) -> list[int]:
    return [x for kind, x in sample["segments"] if kind == "ecg"]


def test_no_arm_reads_the_next_ecg(queries):
    qs, q = queries
    nxt_rows = set(qs.row[q.next_study].tolist())
    partner = qs.swap_partners()
    for i in range(len(qs)):
        for kwargs in ({}, {"partner": int(partner[i])}, {"dates_only": True}):
            rows = rows_of(qs.sample(i, 3, **kwargs))
            assert -1 not in rows and not nxt_rows & set(rows)


def test_swap_keeps_the_offsets_and_replaces_the_earlier_recordings(queries):
    qs, _ = queries
    partner = qs.swap_partners()
    for i in range(len(qs)):
        own, swapped = qs.sample(i, 3), qs.sample(i, 3, partner=int(partner[i]))
        assert "Machine read" not in text_of(own)
        assert text_of(own) == text_of(swapped)  # same offsets, count and question
        assert rows_of(own)[0] == rows_of(swapped)[0]  # the query recording stays
        assert set(rows_of(own)[1:]).isdisjoint(rows_of(swapped)[1:])


def test_without_history_only_the_history_block_is_missing(queries):
    qs, _ = queries
    for i in range(len(qs)):
        k0, own = text_of(qs.sample(i, 0)), text_of(qs.sample(i, 3))
        assert "Earlier ECGs" not in k0 and "Earlier ECGs" in own
        assert own.endswith(k0[k0.index("\n\nQuestion:") :])


def test_dates_only_keeps_the_offsets_and_drops_the_earlier_recordings(queries):
    qs, _ = queries
    for i in range(len(qs)):
        own, dates = qs.sample(i, 3), qs.sample(i, 3, dates_only=True)
        assert rows_of(dates) == rows_of(own)[:1]  # the query only
        assert text_of(dates) == text_of(own)  # same offsets, count and question
