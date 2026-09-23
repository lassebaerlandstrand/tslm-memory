"""Score every arm for one seed on one role and write one JSON line per query.

Arms (adapter, condition, earlier ECGs, control):
- S-none, T-none: the `none` adapter, no earlier ECGs;
- S-none-qswap: S-none with the query recording taken from a same-stratum partner (gate P2);
- S-k0, T-k0: the `ctx` adapter with no earlier ECGs;
- S-own, T-own: the `ctx` adapter with up to 3 own earlier ECGs;
- S-swap, T-swap: the same with a same-stratum partner's earlier ECGs at the query's own offsets;
- T-oracle: T-own plus the next ECG's machine read (a diagnostic, not a gate).
Usage: evaluate.py seed {val,test}
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tslm_context import reader
from tslm_context.samples import Queries
from tslm_context.waves import DATA

BATCH = 32
ARMS = {
    "S-none": ("none", "signal", 0, None),
    "T-none": ("none", "text", 0, None),
    "S-none-qswap": ("none", "signal", 0, "query"),
    "S-k0": ("ctx", "signal", 0, None),
    "T-k0": ("ctx", "text", 0, None),
    "S-own": ("ctx", "signal", 3, None),
    "T-own": ("ctx", "text", 3, None),
    "S-swap": ("ctx", "signal", 3, "history"),
    "T-swap": ("ctx", "text", 3, "history"),
    "T-oracle": ("ctx", "text", 3, "oracle"),
}


def sample(q: Queries, i: int, world: str, k: int, control: str | None, partner: int) -> dict:
    return q.sample(
        i,
        world,
        k,
        partner=partner if control == "history" else None,
        oracle=control == "oracle",
        query_from=partner if control == "query" else None,
    )


def main(seed: int, role: str) -> None:
    if role == "test":
        # The gates are fixed before the test set is scored: gates.json must be committed as is.
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "gates.json"], check=False)
        clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "gates.json"], check=False)
        assert tracked.returncode == 0 and clean.returncode == 0, "commit gates.json first"
    q = Queries(role)
    partner = q.swap_partners(seed=0)
    assert (partner >= 0).all() and (partner != np.arange(len(q))).all()
    out = Path("runs/eval") / role
    out.mkdir(parents=True, exist_ok=True)
    (out / "swap.json").write_text(json.dumps(partner.tolist()))
    table = torch.as_tensor(np.load(DATA / "emb.npy"), device="cuda")
    adapter = reader.Adapter().cuda().eval()
    adapter.load_state_dict(torch.load(f"runs/stage1_s{seed}.pt"))
    for arm in ("none", "ctx"):
        model = reader.load(checkpoint=f"runs/train/{arm}_s{seed}/model.pt")
        for name, (a, world, k, control) in ARMS.items():
            if a != arm:
                continue
            t = time.time()
            recs = []
            for s in range(0, len(q), BATCH):
                chunk = range(s, min(s + BATCH, len(q)))
                batch = [sample(q, i, world, k, control, int(partner[i])) for i in chunk]
                ps = reader.p_yes(model, adapter, table, batch)
                for i, p in zip(chunk, ps, strict=True):
                    r = q.q.iloc[i]
                    recs.append(
                        {
                            "i": i,
                            "subject": int(r.subject_id),
                            "finding": r.finding,
                            "cell": r.cell,
                            "stratum": q.stratum(i),
                            "y": q.label(i),
                            "n_hist": len(r.earlier),
                            "p_yes": p,
                        }
                    )
            path = out / f"{name}_s{seed}.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in recs))
            print(f"{role} {name} s{seed}: n {len(recs)} {time.time() - t:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
