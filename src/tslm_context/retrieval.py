"""Retrieval on top of memory: similar people's ECGs, each stored with what followed it.

The pool is the training subjects' query ECGs with their next ECG (scripts/build_pool.py). A
neighbour item shows the similar ECG and its next ECG as 16 tokens each (text condition: both
machine reads too) and the gap between them. Neighbours never include the query's own subject,
and the pool holds no test or validation subject. The shuffled control keeps each neighbour's ECG
and swaps what followed for that of another pool entry in the same state for the finding asked
about, so it keeps the base rate and breaks the pairing.
"""

import numpy as np
import pandas as pd
import torch

from tslm_context.samples import Queries, machine_read, when
from tslm_context.waves import DATA


class WithNeighbours(Queries):
    def __init__(self, role: str):
        super().__init__(role)
        nb = np.load(DATA / f"neighbours_{role}.npz")
        by_query = dict(zip(nb["query_study"], nb["pool_study"], strict=True))
        self.nb = np.stack([by_query[s] for s in self.q.cur_study])  # pool cur studies [n, K]
        q = pd.read_parquet(DATA / "queries.parquet")
        pool = q[q.role == "train"].set_index("cur_study")
        self.pool_next, self.pool_gap = pool.next_study, pool.gap_days
        nb_subject = pool.subject_id.loc[self.nb.ravel()].to_numpy().reshape(self.nb.shape)
        assert not (nb_subject == self.q.subject_id.to_numpy()[:, None]).any()
        assert role == "train" or set(pool.subject_id).isdisjoint(self.q.subject_id)
        nxt = pd.read_parquet(DATA / "emb_next.parquet").set_index("study_id").row
        # Next ECGs live in a second table appended after the query and history embeddings.
        self.offset = int(self.row.max()) + 1
        self.next_row = nxt[nxt >= 0] + self.offset
        entries = np.array(sorted(set(self.nb.ravel())))
        now = {s: set(self.ecgs.loc[s, "labels"]) for s in entries}
        self.pool_state = {
            (f, has): np.array([e for e in entries if (f in now[e]) == has])
            for f in self.q.finding.unique()
            for has in (False, True)
        }
        self.pool_now = now

    @staticmethod
    def table() -> torch.Tensor:
        emb = np.concatenate([np.load(DATA / "emb.npy"), np.load(DATA / "emb_next.npy")])
        return torch.as_tensor(emb, device="cuda")

    def shuffled_next(self, i: int, rng: np.random.Generator) -> list[int]:
        """For each neighbour of query i, the next ECG of another neighbour-pool entry in the same
        state (finding present or not) for the finding asked about."""
        f = self.q.finding[i]
        out = []
        for s in self.nb[i]:
            same = self.pool_state[(f, f in self.pool_now[s])]
            pick = s
            while pick == s:
                pick = same[rng.integers(len(same))]
            out.append(int(pick))
        return out

    def sample(
        self,
        i: int,
        world: str,
        k: int,
        k_nb: int = 0,
        next_from: list[int] | None = None,
        **kwargs,
    ) -> dict:
        s = super().sample(i, world, k, **kwargs)
        if not k_nb:
            return s
        seg = [("text", "\n\nECGs of other patients similar to this one, with their next ECG:")]
        for j, e in enumerate(self.nb[i][:k_nb]):
            src = int(e if next_from is None else next_from[j])
            nxt = int(self.pool_next[src])
            seg.append(("text", "\n- similar ECG:"))
            seg += self._record(int(e), world)
            seg.append(("text", f" Its next ECG, {when(self.pool_gap[src])} later:"))
            seg.append(("ecg", int(self.next_row[nxt])))
            if world == "text":
                seg.append(("text", machine_read(self.ecgs.loc[nxt])))
        s["segments"] = s["segments"][:-1] + seg + s["segments"][-1:]
        return s
