"""Exact cosine top-k over the index with the query's own patient excluded."""

import numpy as np
import torch


def neighbours(
    q_emb: np.ndarray,
    q_patient: np.ndarray,
    idx_emb: np.ndarray,
    idx_patient: np.ndarray,
    k: int = 32,
    overfetch: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k index rows per query, never the query's own patient and at most one row per patient.

    Returns (rows [Q, k], sims [Q, k]); rows are -1 where fewer than k qualify.
    """
    idx = torch.as_tensor(idx_emb, device="cuda")
    rows_out = np.full((len(q_emb), k), -1, dtype=np.int64)
    sims_out = np.zeros((len(q_emb), k), dtype=np.float32)
    for s in range(0, len(q_emb), 1024):
        q = torch.as_tensor(q_emb[s : s + 1024], device="cuda")
        sims, rows = (q @ idx.T).topk(min(overfetch, len(idx_emb)), dim=1)
        sims, rows = sims.cpu().numpy(), rows.cpu().numpy()
        for i in range(len(rows)):
            seen = {q_patient[s + i]}
            j = 0
            for r, sim in zip(rows[i], sims[i], strict=True):
                if idx_patient[r] in seen:
                    continue
                seen.add(idx_patient[r])
                rows_out[s + i, j], sims_out[s + i, j] = r, sim
                j += 1
                if j == k:
                    break
    return rows_out, sims_out
