"""Day entries: Chronos-2 patch embeddings of each day, written once per split, plus statistics
computed by code when a split is opened."""

import argparse

import numpy as np
import torch
from chronos import Chronos2Pipeline

from tslm_memory.lcl import DATA, SLOTS, load

N_STATS = 7


@torch.no_grad()
def chronos(pipe: Chronos2Pipeline, days: np.ndarray, batch: int = 4096) -> np.ndarray:
    """[N, 48] -> [N, 3, 768] fp16, one vector per 16-reading patch. Chronos-2 takes NaN as
    missing and gives every day its own scale, so the level has to come from the statistics."""
    embeds, _ = pipe.embed(torch.from_numpy(days)[:, None, :], batch_size=batch)
    # Each output is [1, patches + 2, d]: the patches, then [REG], then a future patch.
    return torch.stack([x[0, :-2] for x in embeds]).half().numpy()


# Raw statistics of a day, pooled into weeks and months by the shared rules: total kWh, peak
# half hour in kWh, the time of day of the peak, missing readings. Missing readings count as 0.
RULES = ["sum", "max", 1, "sum"]


def raw_stats(days: np.ndarray) -> np.ndarray:
    """[..., 48] -> [..., 4] by RULES' columns."""
    x = np.nan_to_num(days)
    out = [x.sum(-1), x.max(-1), x.argmax(-1), np.isnan(days).sum(-1)]
    return np.stack(out, -1).astype(np.float32)


def features(raw: np.ndarray, n_days: np.ndarray | int) -> np.ndarray:
    """[..., 4] raw statistics of a day, week or month -> [..., N_STATS] on the day's scale: the
    mean daily total over 10 kWh, the log of the whole total, the peak half hour as kWh and log,
    its time of day as sin and cos, the missing share."""
    n = np.asarray(n_days)
    total, peak = raw[..., 0], raw[..., 1]
    angle = 2 * np.pi * raw[..., 2] / SLOTS
    out = [total / 10 / n, np.log1p(total), peak, np.log1p(peak), np.sin(angle), np.cos(angle)]
    out.append(raw[..., 3] / (SLOTS * n))
    return np.stack(out, -1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    for split in args.splits:
        h = load(split)
        out = np.lib.format.open_memmap(
            DATA / f"{split}_chronos.npy", "w+", np.float16, (int(h.offset[-1]), 3, 768)
        )
        for d, at in zip(h.days, h.offset[:-1], strict=True):
            out[at : at + len(d)] = chronos(pipe, np.array(d, np.float32))
        out.flush()
        print(split, "entries", out.shape)


if __name__ == "__main__":
    main()
