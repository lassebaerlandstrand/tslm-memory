"""Low Carbon London smart meters: parse the half-hourly CSV into days of 48 readings and split
households into train, validation and test.

Household ids are pseudonymous person data. They are only used to group rows and are not stored:
households are numbered in sorted id order."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv

from tslm_memory.questions import Households, Prices

SLOTS = 48
DATA = Path("data/lcl")
SPLITS = {"train": 1000, "val": 200, "test": 200}
# Price questions about events from 1 Oct 2013 are for validation and test only. The schedule is
# the same for every time-of-use household, so a household split alone would let a model learn
# it by date.
HOLDOUT = 273


def parse(path: Path) -> tuple[np.ndarray, ...]:
    """Rows (household, day, slot, kWh) from the CSV. Day counts from 1970-01-01.

    A reading's timestamp ends its half hour: the first reading of a day is at 00:30 and the
    last at 00:00 of the next day, so 30 minutes are taken off before the day and slot."""
    opts = csv.ConvertOptions(column_types={"KWH/hh (per half hour) ": pa.string()})
    reader = csv.open_csv(
        path, convert_options=opts, read_options=csv.ReadOptions(block_size=1 << 26)
    )
    ids: dict[str, int] = {}
    tou: dict[int, bool] = {}
    parts = []
    for batch in reader:
        lcl = batch.column(0).dictionary_encode()
        local = np.array([ids.setdefault(s, len(ids)) for s in lcl.dictionary.to_pylist()])
        hh = local[lcl.indices.to_numpy()]
        u, at = np.unique(hh, return_index=True)
        is_tou = pc.equal(batch.column(1), "ToU").to_numpy(zero_copy_only=False)
        tou.update(zip(u.tolist(), is_tou[at].tolist(), strict=True))
        sec = batch.column(2).cast(pa.timestamp("s")).cast(pa.int64()).to_numpy() - 1800
        text = pc.utf8_trim_whitespace(batch.column(3))
        text = pc.if_else(pc.equal(text, "Null"), None, text)
        kwh = pc.cast(text, pa.float32()).to_numpy(zero_copy_only=False)
        parts.append(
            (
                hh.astype(np.int32),
                (sec // 86400).astype(np.int32),
                (sec % 86400 // 1800).astype(np.int8),
                kwh.astype(np.float32),
            )
        )
    hh, day, slot, kwh = (np.concatenate(c) for c in zip(*parts, strict=True))
    # Number households in sorted id order, so the numbering does not depend on file order.
    rank = np.empty(len(ids), np.int32)
    rank[np.argsort(list(ids))] = np.arange(len(ids))
    is_tou = np.zeros(len(ids), bool)
    for h, t in tou.items():
        is_tou[rank[h]] = t
    return rank[hh], day, slot, kwh, is_tou


def to_days(
    hh: np.ndarray, day: np.ndarray, slot: np.ndarray, kwh: np.ndarray, out: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write every household's calendar days, first to last reading, as [days, 48] float32 with
    NaN for missing readings. Returns per household the first row, the day count, the first day."""
    n = hh.max() + 1
    first = np.full(n, np.iinfo(np.int32).max, np.int64)
    last = np.full(n, -1, np.int64)
    np.minimum.at(first, hh, day)
    np.maximum.at(last, hh, day)
    count = last - first + 1
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    days = np.lib.format.open_memmap(out, "w+", np.float32, (int(count.sum()), SLOTS))
    days[:] = np.nan
    # Duplicate timestamps keep the later row.
    days[start[hh] + day - first[hh], slot] = kwh
    days.flush()
    return start, count, first


def tariffs(path: Path) -> tuple[np.ndarray, int]:
    """The 2013 dynamic price schedule as [days, 48] codes (0 normal, 1 low, 2 high) and its
    first day. Its timestamps start their half hour."""
    t = pd.read_excel(path)
    code = t["Tariff"].map({"Normal": 0, "Low": 1, "High": 2}).to_numpy(np.int8)
    stamp = t["TariffDateTime"].to_numpy("datetime64[m]").astype(np.int64)
    first = int(stamp.min() // 1440)
    grid = np.zeros((int(stamp.max() // 1440) - first + 1, SLOTS), np.int8)
    grid[stamp // 1440 - first, stamp % 1440 // 30] = code
    return grid, first


def split(count: np.ndarray, tou: np.ndarray, min_days: int, seed: int = 0) -> dict:
    """Households with at least min_days days, split at random with the time-of-use share kept."""
    rng = np.random.default_rng(seed)
    out = {s: [] for s in SPLITS}
    total = sum(SPLITS.values())
    for group in [tou, ~tou]:
        pool = rng.permutation(np.flatnonzero(group & (count >= min_days)))
        share = group[count >= min_days].mean()
        at = 0
        for s, n in SPLITS.items():
            k = round(n * share)
            out[s].extend(pool[at : at + k].tolist())
            at += k
    assert sum(len(v) for v in out.values()) == total
    return {s: np.sort(v) for s, v in out.items()}


def load(split: str) -> Households:
    meta = np.load(DATA / "meta.npz")
    days = np.load(DATA / "days.npy", mmap_mode="r")
    ids = meta[split]
    rows = [slice(meta["start"][i], meta["start"][i] + meta["count"][i]) for i in ids]
    grid, first = meta["tariff"], int(meta["tariff_first"])
    asked = (first, first + HOLDOUT) if split == "train" else (first + HOLDOUT, first + len(grid))
    prices = Prices(grid, first, meta["tou"][ids], *asked)
    return Households([days[r] for r in rows], meta["first"][ids], prices)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DATA / "raw/CC_LCL-FullData.csv")
    ap.add_argument("--min-days", type=int, default=300)
    args = ap.parse_args()
    hh, day, slot, kwh, tou = parse(args.csv)
    print(
        f"{len(hh)} rows, {tou.size} households ({tou.sum()} time-of-use), "
        f"{np.isnan(kwh).sum()} null readings"
    )
    start, count, first = to_days(hh, day, slot, kwh, DATA / "days.npy")
    del hh, day, slot, kwh
    grid, grid_first = tariffs(DATA / "raw/Tariffs.xlsx")
    days = np.load(DATA / "days.npy", mmap_mode="r")
    print(
        f"days {days.shape}, per household median {np.median(count):.0f} "
        f"(10%: {np.percentile(count, 10):.0f}, 90%: {np.percentile(count, 90):.0f})"
    )
    parts = split(count, tou, args.min_days)
    np.savez(
        DATA / "meta.npz",
        start=start,
        count=count,
        first=first,
        tou=tou,
        tariff=grid,
        tariff_first=grid_first,
        **parts,
    )
    for s, v in parts.items():
        print(f"{s}: {len(v)} households, {tou[v].sum()} time-of-use, {count[v].sum()} days")


if __name__ == "__main__":
    main()
