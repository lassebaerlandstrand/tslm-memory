import numpy as np

from tslm_memory.report import bootstrap


def test_bootstrap_brackets_the_mean():
    v = (np.random.default_rng(1).random(400) < 0.6).astype(float)
    lo, hi = bootstrap(v, np.arange(400) % 40)
    assert lo < v.mean() < hi and hi - lo < 0.2
