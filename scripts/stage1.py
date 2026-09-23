"""Stage 1: train the MLP that turns a record's ECGFounder embedding into 16 tokens, with the LLM
and its LoRA frozen, by generating the record's own machine read from the tokens.

Trains on records of training subjects (buckets 2-9); checks on records of validation subjects
(bucket 1) with greedy decoding. Pass criteria (gate P1), written to runs/stage1_s{seed}.json:
- per finding, balanced accuracy of the decoded labels >= a logistic probe on the same frozen
  embedding minus 0.05;
- heart rate decoded from the tokens has R^2 >= 0.8;
- decoding from another record's tokens gives balanced accuracy <= 0.55 per finding.
Usage: stage1.py seed
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score

from tslm_context import reader
from tslm_context.mimic import FINDINGS
from tslm_context.samples import Queries, machine_read
from tslm_context.waves import DATA

EPOCHS, BATCH, LR = 5, 64, 1e-3


def records(q: Queries) -> np.ndarray:
    studies = set(q.q.cur_study) | {s for h in q.q.earlier for s in h}
    return np.array(sorted(studies))


def item(q: Queries, study: int) -> dict:
    target = machine_read(q.ecgs.loc[study])[len(" Machine read: ") :]
    seg = [("text", "ECG:"), ("ecg", int(q.row[study])), ("text", " Machine read:")]
    return {"segments": seg, "target": target}


def decoded(text: str) -> tuple[set[str], float]:
    statement = text.split(" Machine read:")[-1].split(". Measures:")[0]
    rate = re.search(r"rate (\d+) bpm", text)
    return {x.strip() for x in statement.split(";")}, float(rate.group(1)) if rate else np.nan


def main(seed: int) -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = Queries("train"), Queries("val")
    rec_tr, rec_va = records(tr), records(va)
    table = torch.as_tensor(np.load(DATA / "emb.npy"), device="cuda")
    model = reader.load()
    adapter = reader.Adapter().cuda()
    opt = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(rec_tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.03)
    t, step = time.time(), 0
    for epoch in range(EPOCHS):
        order = rng.permutation(rec_tr)
        run = []
        for s in range(0, len(order) - BATCH + 1, BATCH):
            loss = reader.lm_loss(
                model, adapter, table, [item(tr, x) for x in order[s : s + BATCH]]
            )
            loss.backward()
            if step == 0:
                assert all(
                    p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters()
                )
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "of": steps,
                            "loss": float(np.mean(run)),
                            "min": (time.time() - t) / 60,
                            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                        }
                    ),
                    flush=True,
                )
                run = []

    adapter.eval()
    out = Path("runs")
    torch.save(adapter.state_dict(), out / f"stage1_s{seed}.pt")
    shuffled = rng.permutation(rec_va)
    res = {"seed": seed, "n_train_records": len(rec_tr), "n_val_records": len(rec_va)}
    for name, source in (("own", rec_va), ("swapped", shuffled)):
        texts = []
        for s in range(0, len(rec_va), 64):
            batch = [item(va, x) for x in source[s : s + 64]]
            texts += reader.generate(model, adapter, table, batch)
        dec = [decoded(x) for x in texts]
        for f in FINDINGS:
            y = np.array([f in va.ecgs.loc[x, "labels"] for x in rec_va])
            pred = np.array([f in d for d, _ in dec])
            res[f"bacc {name} | {f}"] = balanced_accuracy_score(y, pred)
        hr = va.ecgs.loc[rec_va, "hr"].to_numpy()
        got = np.array([r for _, r in dec])
        ok = ~np.isnan(hr) & ~np.isnan(got)
        res[f"hr r2 {name}"] = r2_score(hr[ok], got[ok])
        res[f"hr decoded share {name}"] = float(ok.mean())
    emb = table.cpu().numpy()
    for f in FINDINGS:
        y_tr = np.array([f in tr.ecgs.loc[x, "labels"] for x in rec_tr])
        y_va = np.array([f in va.ecgs.loc[x, "labels"] for x in rec_va])
        probe = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        probe.fit(emb[tr.row[rec_tr].to_numpy()], y_tr)
        res[f"bacc probe | {f}"] = balanced_accuracy_score(
            y_va, probe.predict(emb[va.row[rec_va].to_numpy()])
        )
    res["pass"] = bool(
        all(res[f"bacc own | {f}"] >= res[f"bacc probe | {f}"] - 0.05 for f in FINDINGS)
        and res["hr r2 own"] >= 0.8
        and all(res[f"bacc swapped | {f}"] <= 0.55 for f in FINDINGS)
    )
    (out / f"stage1_s{seed}.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, round(v, 3) if isinstance(v, float) else v)


if __name__ == "__main__":
    main(int(sys.argv[1]))
