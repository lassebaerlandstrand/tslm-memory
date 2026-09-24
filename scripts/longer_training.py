"""Exploratory, after the gates: does the gain from own history grow with training?

Trains the none and ctx arms of one seed for twice the epochs into runs/longer/, then scores
S-none, S-own and S-swap on the test patients. Writes runs/longer/{arm}_s{seed}.jsonl.
Usage: longer_training.py seed epochs
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

import train
from evaluate import ARMS, sample
from tslm_context import reader
from tslm_context.samples import Queries
from tslm_context.waves import DATA

BATCH = 32


def main(seed: int, epochs: int) -> None:
    out = Path("runs/longer")
    for arm in ("none", "ctx"):
        if not (out / f"{arm}_s{seed}" / "model.pt").exists():
            train.main(arm, seed, epochs=epochs, out=out / f"{arm}_s{seed}")
    q = Queries("test")
    partner = q.swap_partners(seed=0)
    table = torch.as_tensor(np.load(DATA / "emb.npy"), device="cuda")
    for arm, names in (("none", ["S-none"]), ("ctx", ["S-own", "S-swap"])):
        model = reader.load(checkpoint=str(out / f"{arm}_s{seed}" / "model.pt"))
        adapter = reader.Adapter().cuda().eval()
        adapter.load_state_dict(torch.load(out / f"{arm}_s{seed}" / "adapter.pt"))
        for name in names:
            _, world, k, control = ARMS[name]
            recs = []
            for s in range(0, len(q), BATCH):
                chunk = range(s, min(s + BATCH, len(q)))
                batch = [sample(q, i, world, k, control, int(partner[i])) for i in chunk]
                ps = reader.p_yes(model, adapter, table, batch)
                recs += [
                    {"i": i, "stratum": q.stratum(i), "cell": q.q.cell[i], "y": q.label(i), "p_yes": p}
                    for i, p in zip(chunk, ps, strict=True)
                ]
            path = out / f"{name}_s{seed}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))
            print(f"{name} s{seed} ({epochs} epochs): n {len(recs)}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
