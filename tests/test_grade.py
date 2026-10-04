import numpy as np

from tslm_memory_poc.grade import auroc_within, bootstrap, ci
from tslm_memory_poc.waves import unusable


def test_persistence_scores_half_within_strata():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    strata = np.array(["a|0"] * 4 + ["a|1"] * 4)
    now = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    assert auroc_within(y, now, strata) == 0.5


def test_bootstrap_resamples_within_cells():
    cells = np.array(["x"] * 30 + ["y"] * 10)
    seen = []
    bootstrap(lambda idx: seen.append(idx) or 0.0, cells, n=20)
    for idx in seen:
        assert (cells[idx] == "x").sum() == 30 and (cells[idx] == "y").sum() == 10


def test_seed_spread_widens_the_interval():
    boots = np.random.default_rng(0).normal(0.03, 0.01, 2000)
    lo1, hi1 = ci(0.03, boots, [])
    lo2, hi2 = ci(0.03, boots, [0.0, 0.03, 0.06])
    assert lo2 < lo1 < 0.03 < hi1 < hi2


def test_unusable_records():
    ok = np.random.default_rng(0).normal(0, 1, (5000, 12))
    assert not unusable(ok)
    missing = ok.copy()
    missing[:, 3] = np.nan
    assert unusable(missing)
    assert unusable(np.zeros((5000, 12)))
    dropout = ok.copy()
    dropout[:100, :] = np.nan
    assert not unusable(dropout)
