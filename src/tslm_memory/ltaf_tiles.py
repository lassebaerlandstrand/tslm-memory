"""Base entries of every LTAF record: back to back 2.5 s two-lead tiles from the record's start,
a grid every benchmark window starts on (windows start on 5 s). Each tile gets frozen Chronos-2
vectors and statistics computed by code, stored as sums so that coarser levels pool exactly.
Written once to data/tshaystack/base_chronos.npy, base_chronos_stats.npy and the index."""

import argparse
import time

import numpy as np
import torch
from chronos import Chronos2Pipeline
from scipy import signal as ss

from tslm_memory.ltaf import HZ, RECORDS, ROOT, gains, signal

TILE = 5 * HZ // 2  # samples per base entry
# Per lead the sum, sum of squares, min and max in mV; then detected R-peaks, the sum of their
# RR intervals in s and of their squares (the RR ending at each peak).
STATS = ["sum0", "sum1", "sq0", "sq1", "min0", "min1", "max0", "max1", "beats", "rr", "rr2"]


def detect(x: np.ndarray) -> np.ndarray:
    """R-peak samples: 8-25 Hz band, squared slope summed over leads, 80 ms average, peaks above
    0.4 of the 98th percentile of their 8 s block and at least 0.3 s apart."""
    b, a = ss.butter(2, [8 / (HZ / 2), 25 / (HZ / 2)], "band")
    y = ss.filtfilt(b, a, x, axis=0)
    e = (np.diff(y, axis=0, prepend=y[:1]) ** 2).sum(1)
    w = int(0.08 * HZ)
    e = np.convolve(e, np.ones(w) / w, mode="same")
    blk = 8 * HZ
    n = len(e) // blk
    q = np.percentile(e[: n * blk].reshape(n, blk), 98, axis=1)
    thr = np.concatenate([np.repeat(q, blk), np.full(len(e) - n * blk, q[-1])]) * 0.4
    return ss.find_peaks(e, height=thr, distance=int(0.3 * HZ))[0]


def tile_stats(x: np.ndarray, n: int) -> np.ndarray:
    """[n, len(STATS)] for the first n tiles of x [samples, 2] mV."""
    t = x[: n * TILE].reshape(n, TILE, 2)
    out = np.zeros((n, len(STATS)), np.float32)
    out[:, 0:2], out[:, 2:4] = t.sum(1), (t**2).sum(1)
    out[:, 4:6], out[:, 6:8] = t.min(1), t.max(1)
    p = detect(x[: n * TILE])
    rr = np.diff(p, prepend=p[0]) / HZ
    k = p // TILE
    np.add.at(out[:, 8], k, 1.0)
    np.add.at(out[:, 9], k[1:], rr[1:])
    np.add.at(out[:, 10], k[1:], rr[1:] ** 2)
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
    records = sorted(p.stem for p in RECORDS.glob("*.dat"))
    count = np.array([len(signal(r)) // TILE for r in records])
    offset = np.concatenate([[0], np.cumsum(count)[:-1]])
    name = ROOT / "base_chronos"
    out = np.lib.format.open_memmap(f"{name}.npy", "w+", np.float16, (int(count.sum()), 4, 768))
    stats = np.lib.format.open_memmap(
        f"{name}_stats.npy", "w+", np.float32, (int(count.sum()), len(STATS))
    )
    t0 = time.time()
    for r, at, n in zip(records, offset, count, strict=True):
        x = np.asarray(signal(r), np.float32) / gains(r)
        stats[at : at + n] = tile_stats(x, n)
        tiles = x[: n * TILE].reshape(n, TILE, 2).transpose(0, 2, 1)  # [n, 2, TILE]
        for i in range(0, n, args.batch):
            embeds, _ = pipe.embed(
                torch.from_numpy(np.ascontiguousarray(tiles[i : i + args.batch]))
            )
            # Per lead the mean patch embedding and the [REG] token.
            e = torch.stack([torch.cat([v[:, :-2].mean(1), v[:, -2]], -1) for v in embeds])
            out[at + i : at + i + len(e)] = e.view(len(e), 4, 768).half().numpy()
        print(f"{r} {n} tiles, {time.time() - t0:.0f}s", flush=True)
    out.flush()
    stats.flush()
    np.savez(f"{name}_index.npz", records=records, offset=offset, count=count)
    print("base entries", out.shape)


if __name__ == "__main__":
    main()
