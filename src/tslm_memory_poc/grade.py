"""Scoring: AUROC within strata and the bootstrap with seed spread."""

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_within(y: np.ndarray, score: np.ndarray, strata: np.ndarray) -> float:
    """Mean AUROC over strata (finding x current state), so persistence scores 0.5."""
    return float(
        np.mean([roc_auc_score(y[strata == s], score[strata == s]) for s in np.unique(strata)])
    )


def bootstrap(stat, cells: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    """stat(idx) over n resamples drawn with replacement within each cell. Every test subject
    contributes one query, so resampling queries is resampling subjects."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    return np.array(
        [stat(np.concatenate([rng.choice(m, size=len(m)) for m in members])) for _ in range(n)]
    )


def ci(estimate: float, boots: np.ndarray, per_seed: list[float]) -> tuple[float, float]:
    """95 % interval with SE_tot^2 = SE_boot^2 + Var(per-seed estimates) / n_seeds."""
    se2 = np.var(boots, ddof=1)
    if len(per_seed) > 1:
        se2 += np.var(per_seed, ddof=1) / len(per_seed)
    half = 1.96 * np.sqrt(se2)
    return float(estimate - half), float(estimate + half)
