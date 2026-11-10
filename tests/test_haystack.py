import numpy as np
import torch

from tslm_memory.haystack import (
    N_STATS,
    RULES,
    Base,
    Data,
    duration,
    features,
    hits,
    recall_text,
)
from tslm_memory.haystack_score import correct
from tslm_memory.ltaf import HZ, clock
from tslm_memory.ltaf_tiles import TILE
from tslm_memory.memory import Part, batch, grid_levels, pool_stats


def test_score_follows_the_benchmark():
    # Time ranges at IoU >= 0.25: [0, 10] against [5, 15] is 1/3.
    assert correct("00:00:00.000-00:00:10.000", "00:00:05.000-00:00:15.000", "time_range")
    assert not correct("00:00:00.000-00:00:10.000", "00:00:08.000-00:00:20.000", "time_range")
    assert correct("01:44:48.961-01:44:49.875", "01:44:49-01:44:50", "time_range")
    # Single timestamps within 0.5 s.
    assert correct("00:26:39.531", "00:26:39.900", "timestamp")
    assert not correct("00:26:39.531", "00:26:40.100", "timestamp")
    assert correct("yes", "Yes, it does.", "boolean")
    assert not correct("no", "yes", "boolean")
    assert correct("40", "40", "integer") and not correct("4", "40", "integer")
    assert correct("NSR", "nsr", "category") and not correct("NSR", "AFIB", "category")
    # Their category rule takes an empty answer as right; here it is wrong.
    assert not correct("NSR", "", "category")


def test_clock_matches_the_benchmark_format():
    assert clock(52815) == "00:06:52.617"
    assert duration(900 * 128) == "15 minutes" and duration(7200 * 128) == "2 hours"


def test_recall_names_entries_in_time_order_and_merges_overlapping_ones():
    start = list(np.arange(6, dtype=np.float32) * 10)
    end = [x + 10 for x in start]
    assert (
        recall_text(start, end, [""] * 6, [4, 1, 2])
        == "00:00:10.000-00:00:30.000; 00:00:40.000-00:00:50.000"
    )
    # A coarser entry over finer ones merges with them.
    got = recall_text([0, 10, 0, 50.0], [10, 20, 40, 60.0], [""] * 4, [3, 1, 2])
    assert got == "00:00:00.000-00:00:40.000; 00:00:50.000-00:01:00.000"
    end = list(start)
    end[0] = 75.5
    names = ["AFIB", "V", "", "", "", ""]
    assert (
        recall_text(start, end, names, [1, 0]) == "AFIB 00:00:00.000-00:01:15.500; V 00:00:10.000"
    )


def test_chunk_tokens_span_their_slots():
    from types import SimpleNamespace

    from tslm_memory.haystack import tokens

    start = np.array([0, 10, 20, 30], np.float32)
    entries = Part(torch.zeros(4, 1), start, start + 10, np.zeros(4, bool), level=np.zeros(4, int))
    items = Part(torch.zeros(0, 1), start[:0], start[:0], np.zeros(0, bool))
    b = batch([entries], [items], [True], (1, 1, 0, 1), "cpu")
    # Chunk 0 holds slots 1 and 2 (entries 0 and 1), chunk 1 is empty, chunk 2 holds entry 3.
    chunk_of = torch.zeros(1, 3, 5, dtype=torch.bool)
    chunk_of[0, 0, [1, 2]] = True
    chunk_of[0, 2, 4] = True
    lo, hi, names = tokens(b, SimpleNamespace(chunk_of=chunk_of))
    assert lo[0].tolist() == [0, 0, 30] and hi[0].tolist() == [20, 0, 40] and names == [[""] * 3]
    # An empty bank: no slots, so nothing to span.
    none = Part(torch.zeros(0, 1), start[:0], start[:0], np.zeros(0, bool), level=np.zeros(0, int))
    empty = batch([none], [items], [True], (1, 1, 0, 1), "cpu")
    lo, hi, _ = tokens(empty, SimpleNamespace(chunk_of=torch.zeros(1, 3, 1, dtype=torch.bool)))
    assert lo.shape == (1, 0)


def test_ecg_statistics_pool_into_the_features_of_the_whole_span():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, (16 * TILE, 2)).astype(np.float32)
    p = np.sort(rng.choice(len(x), 40, replace=False))
    raw = np.zeros((16, N_STATS), np.float32)
    t = x.reshape(16, TILE, 2)
    raw[:, 0:2], raw[:, 2:4], raw[:, 4:6], raw[:, 6:8] = t.sum(1), (t**2).sum(1), t.min(1), t.max(1)
    # Each peak's RR reaches back to the peak before it, here one before the span.
    rr = np.diff(p, prepend=p[0] - 100) / HZ
    np.add.at(raw[:, 8], p // TILE, 1)
    np.add.at(raw[:, 9], p // TILE, rr)
    np.add.at(raw[:, 10], p // TILE, rr**2)
    g = grid_levels(16, 4, 3)
    f = features(pool_stats(raw, RULES, g), g[:, 2] - g[:, 1] + 1)[-1]
    want = [*x.mean(0), *x.std(0), *x.min(0), *x.max(0), 40 / (16 * TILE / HZ), rr.mean(), rr.std()]
    assert np.allclose(f, want, atol=1e-4)


def test_coarse_entries_are_evidence_only_when_half_inside_the_evidence():
    n = 64 * TILE
    g = grid_levels(64, 4, 4)
    j = np.arange(64)
    groups = np.concatenate([np.stack([np.zeros_like(j), j, j], 1), g])
    a, e, level = groups[:, 1] * TILE, (groups[:, 2] + 1) * TILE, groups[:, 0]
    # A 30 s bout from base 4 to 15 plus a beat at the start of base 40.
    ev = [("bout", 4 * TILE, 16 * TILE), ("beat", 40 * TILE, 40 * TILE)]
    hit = hits(a, e, level, ev, n)
    on = {(int(lv), int(x // TILE)) for lv, x in zip(level[hit], a[hit], strict=True)}
    base = {(0, k) for k in [*range(4, 16), 40]}
    # Level 1 groups 1 to 3 lie inside the bout; level 2 group 0 (base 0 to 15) is 3/4 inside;
    # level 3 (base 0 to 63) and every group around the beat are not.
    assert on == base | {(1, 4), (1, 8), (1, 12), (2, 0)}


def test_tiles_are_10_s_entries_with_levels_pooled_from_the_2_5_s_rows():
    rng = np.random.default_rng(0)
    base = Base.__new__(Base)
    base.x = rng.normal(size=(70, 1, 3)).astype(np.float16)
    base.raw = np.abs(rng.normal(size=(70, N_STATS))).astype(np.float32)
    base.offset, base.count = {"r": 0}, {"r": 70}
    data = Data.__new__(Data)
    data.base = base
    # 160 s from the record's 6th row: sixteen 10 s, four 40 s and one 160 s entry.
    n = 64 * TILE
    p = data.tiles("r", 6 * TILE, n, [("bout", 0, 8 * TILE)], "cpu")
    assert np.bincount(p.level).tolist() == [16, 4, 1]
    assert p.start[:2].tolist() == [0, 10] and p.end[16] == 40 and p.end[-1] == 160
    want = base.x[6:10].astype(np.float32).mean(0).ravel()
    assert np.allclose(p.vectors[0].float().numpy(), want, atol=1e-3)
    # The bout of 20 s is two 10 s entries, and half of the first 40 s entry.
    assert p.evidence.tolist() == [True, True] + [False] * 14 + [True] + [False] * 4


def test_each_10_s_entry_is_labelled_with_the_rhythm_at_its_centre():
    import pandas as pd

    from tslm_memory.ltaf import RHYTHMS

    data = Data.__new__(Data)
    bouts = pd.DataFrame(
        {"label": ["AFIB", "NSR"], "start": [0, 25 * HZ], "end": [12 * HZ, 60 * HZ]}
    )
    data.events = {("r", 0, 60 * HZ): (bouts, None)}
    start = np.array([0, 10, 20, 30, 40, 50, 0], np.float32)
    end = np.array([10, 20, 30, 40, 50, 60, 40], np.float32)
    level = np.array([0, 0, 0, 0, 0, 0, 1])
    p = Part(torch.zeros(7, 1), start, end, np.zeros(7, bool), level=level)
    # Centres 5, 15, 25, ...: AFIB, none (the gap from 12 s to 25 s), then NSR; the 40 s entry -1.
    want = [RHYTHMS.index("AFIB"), -1] + [RHYTHMS.index("NSR")] * 4 + [-1]
    assert data.rhythm("r", 0, 60 * HZ, p).tolist() == want
