"""Exploratory, after the gates: does the reader gain more from more of its own history?

Scores the trained ctx adapters on the test patients that have three earlier ECGs, signal only,
with the 1, 2 or 3 most recent of them (k = 0 is S-k0). Same patients in every arm, so only k
varies. Writes runs/curve/S-k{k}_s{seed}.jsonl.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tslm_context import reader
from tslm_context.samples import Queries
from tslm_context.waves import DATA

BATCH = 32


def main(seed: int) -> None:
    q = Queries("test")
    ids = np.flatnonzero(q.q.earlier.map(len).to_numpy() == 3)
    table = torch.as_tensor(np.load(DATA / "emb.npy"), device="cuda")
    model = reader.load(checkpoint=f"runs/train/ctx_s{seed}/model.pt")
    adapter = reader.Adapter().cuda().eval()
    adapter.load_state_dict(torch.load(f"runs/train/ctx_s{seed}/adapter.pt"))
    out = Path("runs/curve")
    out.mkdir(parents=True, exist_ok=True)
    for k in (0, 1, 2, 3):
        t, recs = time.time(), []
        for s in range(0, len(ids), BATCH):
            chunk = ids[s : s + BATCH]
            ps = reader.p_yes(model, adapter, table, [q.sample(int(i), "signal", k) for i in chunk])
            recs += [
                {
                    "i": int(i),
                    "stratum": q.stratum(int(i)),
                    "cell": q.q.cell[i],
                    "y": q.label(int(i)),
                    "p_yes": p,
                }
                for i, p in zip(chunk, ps, strict=True)
            ]
        (out / f"S-k{k}_s{seed}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(f"k {k} s{seed}: n {len(recs)} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
