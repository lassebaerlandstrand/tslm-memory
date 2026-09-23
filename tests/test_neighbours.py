import numpy as np
import pytest
import torch

from tslm_context.index import neighbours

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_own_patient_never_returned_and_one_row_per_patient():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    patient = rng.integers(0, 100, size=500)
    rows, _ = neighbours(emb[:50], patient[:50], emb, patient, k=32)
    for i in range(50):
        got = rows[i][rows[i] >= 0]
        assert patient[i] not in set(patient[got])
        assert len(set(patient[got])) == len(got)


def test_query_itself_is_excluded_even_when_in_index():
    emb = np.eye(8, dtype=np.float32)
    patient = np.arange(8)
    rows, sims = neighbours(emb, patient, emb, patient, k=3)
    assert (rows != np.arange(8)[:, None]).all()
    assert (sims < 1).all()
