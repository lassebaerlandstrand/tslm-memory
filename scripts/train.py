"""Stage 2: continue the released LoRA on readout B with the stage-1 MLP frozen.

Two arms see the same samples in the same order with the same condition per sample (signal or
text, 50/50): `none` never sees earlier ECGs; `ctx` sees k of the patient's own earlier ECGs with
k drawn uniformly from 0 to the number available. Loss: cross-entropy of " yes"/" no" at the answer
position. Usage: train.py {none,ctx} seed
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

EPOCHS, BATCH, LR, WD = 2, 16, 1e-4, 0.01


def main(arm: str, seed: int) -> None:
    assert arm in ("none", "ctx")
    out = Path("runs/train") / f"{arm}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    q = Queries("train")
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(len(q)) for _ in range(EPOCHS)])
    # Drawn for every sample in both arms so the two runs consume identical random streams.
    world = np.where(rng.random(len(order)) < 0.5, "signal", "text")
    frac = rng.random(len(order))
    n_hist = q.q.earlier.map(len).to_numpy()[order]
    k = np.floor(frac * (n_hist + 1)).astype(int) if arm == "ctx" else np.zeros(len(order), int)

    torch.manual_seed(seed)
    table = torch.as_tensor(np.load(DATA / "emb.npy"), device="cuda")
    model = reader.load()
    adapter = reader.Adapter().cuda()
    adapter.load_state_dict(torch.load(f"runs/stage1_s{seed}.pt"))
    adapter.requires_grad_(False)
    frozen = [p.detach().clone() for p in adapter.parameters()]
    # reader.load freezes everything; OpenTSLM saves only LoRA tensors that require grad.
    params = [p for n, p in model.llm.named_parameters() if "lora_" in n]
    for p in params:
        p.requires_grad = True
    # Text-world samples with three earlier ECGs reach about 450 tokens; without checkpointing a
    # batch of 16 needs more than 16 GB.
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.train()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
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
            q.sample(int(i), str(w), int(kk))
            for i, w, kk in zip(order[sl], world[sl], k[sl], strict=True)
        ]
        loss = reader.loss(model, adapter, table, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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
    assert all(torch.equal(a, b) for a, b in zip(frozen, adapter.parameters(), strict=True))
    model.store_to_file(str(out / "model.pt"))
    print("saved", out / "model.pt")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
