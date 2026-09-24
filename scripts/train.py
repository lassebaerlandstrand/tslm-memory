"""Stage 2: continue the released LoRA on readout B, training the stage-1 adapter jointly.

The adapter starts from stage 1 in every arm and trains with its own learning rate (with it frozen
the reader stayed near chance on validation: 0.567 against 0.624 jointly).

Three arms see the same samples in the same order with the same condition per sample (signal or
text, 50/50): `none` never sees earlier ECGs; `ctx` sees k of the patient's own earlier ECGs with
k drawn uniformly from 0 to the number available; `ret` sees the same own history as ctx plus 0, 2
or 4 similar ECGs of other training subjects, each with its next ECG (tslm_context.retrieval).
Loss: cross-entropy of " yes"/" no" at the answer position. Usage: train.py {none,ctx,ret} seed
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tslm_context import reader
from tslm_context.retrieval import WithNeighbours
from tslm_context.samples import Queries
from tslm_context.waves import DATA

EPOCHS, BATCH, LR, AD_LR, WD = 2, 16, 1e-4, 1e-3, 0.01


def main(arm: str, seed: int) -> None:
    assert arm in ("none", "ctx", "ret")
    out = Path("runs/train") / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    q = WithNeighbours("train") if arm == "ret" else Queries("train")
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(q)) for _ in range(EPOCHS)])
    # Drawn for every sample in both arms so the two runs consume identical random streams.
    world = np.where(rng.random(len(order)) < 0.5, "signal", "text")
    frac = rng.random(len(order))
    n_hist = q.q.earlier.map(len).to_numpy()[order]
    k = np.floor(frac * (n_hist + 1)).astype(int) if arm != "none" else np.zeros(len(order), int)
    # Drawn after the shared streams, so none and ctx are unchanged by the ret arm.
    k_nb = rng.choice([0, 2, 4], size=len(order)) if arm == "ret" else np.zeros(len(order), int)

    torch.manual_seed(seed)
    table = (
        WithNeighbours.table()
        if arm == "ret"
        else torch.as_tensor(np.load(DATA / "emb.npy"), device="cuda")
    )
    model = reader.load()
    adapter = reader.Adapter().cuda()
    adapter.load_state_dict(torch.load(f"runs/stage1_s{seed}.pt"))
    # reader.load freezes everything; OpenTSLM saves only LoRA tensors that require grad.
    params = [p for n, p in model.llm.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad = True
    # Text-world samples with three earlier ECGs reach about 450 tokens; without checkpointing a
    # batch of 16 needs more than 16 GB.
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.train()
    opt = torch.optim.AdamW(
        [{"params": params, "lr": LR}, {"params": adapter.parameters(), "lr": AD_LR}],
        weight_decay=WD,
    )
    steps = len(order) // BATCH
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm)))
    )
    log = open(out / "log.jsonl", "w")
    t, run = time.time(), []
    for step in range(steps):
        sl = slice(step * BATCH, (step + 1) * BATCH)
        batch = [
            q.sample(int(i), str(w), int(kk), k_nb=int(nb))
            if nb
            else q.sample(int(i), str(w), int(kk))
            for i, w, kk, nb in zip(order[sl], world[sl], k[sl], k_nb[sl], strict=True)
        ]
        loss = reader.loss(model, adapter, table, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params + list(adapter.parameters()), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run.append(loss.item())
        if (step + 1) % 50 == 0 or step + 1 == steps:
            rec = {
                "step": step + 1,
                "of": steps,
                "loss": float(np.mean(run)),
                "min": (time.time() - t) / 60,
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run = []
    model.store_to_file(str(out / "model.pt"))
    torch.save(adapter.state_dict(), out / "adapter.pt")
    print("saved", out / "model.pt")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
