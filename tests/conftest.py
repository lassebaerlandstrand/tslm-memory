import numpy as np
import pandas as pd
import pytest

from tslm_context.mimic import FINDINGS


@pytest.fixture(scope="session")
def ecgs() -> pd.DataFrame:
    """Synthetic canonical table: 2,000 subjects with 6 ECGs each, random findings, some only
    qualified, and a few ECGs too close together to count."""
    rng = np.random.default_rng(0)
    rows = []
    study = 1
    for subject in range(10_000, 12_000):
        t = pd.Timestamp("2150-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
        for _ in range(6):
            t += pd.Timedelta(hours=float(rng.choice([0.5, 30, 400])))
            labels = [f for f in FINDINGS if rng.random() < 0.5]
            possible = [f for f in FINDINGS if f not in labels and rng.random() < 0.05]
            rows.append((subject, study, t, labels, possible, f"files/p{subject}/s{study}/{study}"))
            study += 1
    df = pd.DataFrame(
        rows, columns=["subject_id", "study_id", "ecg_time", "labels", "possible", "path"]
    )
    for m in ["hr", "pr", "qrs", "qt", "qtc", "p_axis", "qrs_axis", "t_axis"]:
        df[m] = rng.normal(100, 10, len(df))
    return df.sort_values(["subject_id", "ecg_time"]).reset_index(drop=True)
