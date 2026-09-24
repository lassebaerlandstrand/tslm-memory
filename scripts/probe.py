"""The frozen reader read by a linear probe (gates_confirm.json).

Exploration after the gates (notes) found that stage 2 removes what the frozen reader holds. Here
the reader is the released OpenTSLM LoRA with the stage-1 adapter of one seed and no stage 2. Its
hidden state after layer 12 at "Answer:" is read by one standardised logistic regression per
finding, C by 5-fold cross-validation on the 8,000 training queries (AUROC within current state).
The S-none and S-own probes are fitted on their own arm; S-swap (a same-stratum partner's earlier
ECGs, as in evaluate.py) is scored with the S-own probe. Writes
runs/eval/{test,confirm}/probe-{arm}_s{seed}.jsonl like evaluate.py. Usage: probe.py seed
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from evaluate import committed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tslm_context import reader
from tslm_context.grade import auroc_within
from tslm_context.samples import Queries
from tslm_context.waves import DATA

LAYER = 12
CS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
ARMS = {"S-none": (0, "S-none"), "S-own": (3, "S-own"), "S-swap": (3, "S-own")}  # k, probe


@torch.no_grad()
def hidden(model, adapter, table, q: Queries, k: int, partner=None) -> np.ndarray:
    out = []
    for s in range(0, len(q), 32):
        ids = range(s, min(s + 32, len(q)))
        batch = [
            q.sample(i, "signal", k, partner=None if partner is None else int(partner[i]))
            for i in ids
        ]
        emb, att = reader.embed_left(model, adapter, table, batch)
        pos = (att.cumsum(dim=1) - 1).clamp(min=0)
        o = model.llm(
            inputs_embeds=emb,
            attention_mask=att,
            position_ids=pos,
            output_hidden_states=True,
            logits_to_keep=1,
        )
        out.append(o.hidden_states[LAYER][:, -1].float().cpu())
    return torch.cat(out).numpy()


def cv_auroc(x: np.ndarray, y: np.ndarray, now: np.ndarray, c: float) -> float:
    s = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        s[va] = LogisticRegression(C=c, max_iter=3000).fit(x[tr], y[tr]).decision_function(x[va])
    return auroc_within(y, s, now)


def fit(x: np.ndarray, q: Queries) -> dict:
    """Per finding: the scaler and the logistic regression with C chosen by 5-fold CV."""
    y = np.array([q.label(i) for i in range(len(q))])
    now = q.q.cell.str[0].to_numpy()
    probes = {}
    for f in q.q.finding.unique():
        m = (q.q.finding == f).to_numpy()
        sc = StandardScaler().fit(x[m])
        xf = sc.transform(x[m])
        c = CS[int(np.argmax([cv_auroc(xf, y[m], now[m], c) for c in CS]))]
        probes[f] = (sc, LogisticRegression(C=c, max_iter=3000).fit(xf, y[m]))
        print(f"probe {f}: C {c}", flush=True)
    return probes


def main(seed: int) -> None:
    # The confirmation gates are fixed before the confirmation set is scored.
    assert committed("gates_confirm.json"), "commit gates_confirm.json first"
    tr = Queries("train")
    table = torch.as_tensor(np.load(DATA / "emb.npy"), device="cuda")
    model = reader.load()
    adapter = reader.Adapter().cuda().eval()
    adapter.load_state_dict(torch.load(f"runs/stage1_s{seed}.pt"))
    probes = {
        arm: fit(hidden(model, adapter, table, tr, k), tr)
        for arm, k in (("S-none", 0), ("S-own", 3))
    }
    for role in ("test", "confirm"):
        ev = Queries(role)
        partner = ev.swap_partners(seed=0)
        out = Path("runs/eval") / role
        out.mkdir(parents=True, exist_ok=True)
        for arm, (k, probe) in ARMS.items():
            t = time.time()
            x = hidden(model, adapter, table, ev, k, partner if arm == "S-swap" else None)
            p = np.zeros(len(ev))
            for f, (sc, lr) in probes[probe].items():
                m = (ev.q.finding == f).to_numpy()
                p[m] = lr.predict_proba(sc.transform(x[m]))[:, 1]
            recs = [ev.result(i, float(p[i])) for i in range(len(ev))]
            (out / f"probe-{arm}_s{seed}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in recs)
            )
            print(f"{role} probe-{arm} s{seed}: n {len(ev)} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
