import numpy as np
import pytest

from tslm_memory.mimic import bucket
from tslm_memory.selection import BUCKETS, CONFIRM, N_HIST, select, to_fetch

PER_CELL = {"test": 2, "val": 1, "train": 3}


@pytest.fixture(scope="module")
def queries(ecgs):
    return select(ecgs, PER_CELL, reserve=0.0)


def test_every_subject_once_and_in_its_role(queries):
    assert queries.subject_id.is_unique
    for role, bk in BUCKETS.items():
        r = queries[queries.role == role]
        assert all(bucket(s) in bk for s in r.subject_id)
        assert (r.groupby(["finding", "cell"]).size() == PER_CELL[role]).all()
        assert r.groupby(["finding", "cell"]).ngroups == 16


def test_cells_that_cannot_fill_raise(ecgs):
    with pytest.raises(AssertionError):
        select(ecgs, {"test": 500, "val": 1, "train": 1}, reserve=0.0)


def test_history_is_own_earlier_and_spaced(ecgs, queries):
    by_study = ecgs.set_index("study_id")
    for r in queries.itertuples():
        t_cur = by_study.ecg_time[r.cur_study]
        prev = t_cur
        assert r.n_hist >= 2
        for j in range(N_HIST):
            s = getattr(r, f"hist_study_{j}")
            if s < 0:
                continue
            assert by_study.subject_id[s] == r.subject_id
            assert (prev - by_study.ecg_time[s]).total_seconds() >= 3600
            prev = by_study.ecg_time[s]
        assert by_study.subject_id[r.next_study] == r.subject_id
        assert by_study.ecg_time[r.next_study] > t_cur


def test_cell_labels_and_no_qualified_mentions(ecgs, queries):
    by_study = ecgs.set_index("study_id")
    for r in queries.itertuples():
        now = r.finding in by_study.labels[r.cur_study]
        then = r.finding in by_study.labels[r.next_study]
        assert r.cell == f"{int(now)}{int(then)}"
        for s in (r.cur_study, r.next_study):
            assert r.finding not in by_study.possible[s]


def test_next_ecgs_are_never_fetched(queries):
    fetched = to_fetch(queries)
    assert not set(queries.next_study) & set(fetched)
    test_subjects = set(queries[queries.role == "test"].subject_id)
    assert np.isin(fetched[: 2 * 16 * PER_CELL["test"]], fetched).all()
    assert test_subjects.isdisjoint(set(queries[queries.role != "test"].subject_id))


def test_confirmation_role_is_fresh_bucket_1(ecgs, queries):
    used = frozenset(queries.subject_id)
    c = select(ecgs, {"confirm": 2}, reserve=0.0, buckets=CONFIRM, exclude=used)
    assert c.subject_id.is_unique and used.isdisjoint(c.subject_id)
    assert all(bucket(s) == 1 for s in c.subject_id)
    assert (c.groupby(["finding", "cell"]).size() == 2).all()
    assert c.groupby(["finding", "cell"]).ngroups == 16
