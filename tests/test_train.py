import numpy as np
import torch

from tslm_memory.encode import N_STATS
from tslm_memory.questions import FAMILIES, Households, Prices, ask
from tslm_memory.train import Split, memory_for

WINDOW = 60


def split() -> Split:
    """Four households whose day entries hold (household, day index) and whose week and month
    entries hold (household, -1 - period row), so the bank can be traced."""
    rng = np.random.default_rng(0)
    days = [
        rng.gamma(2, 0.1, (n, 48)) * rng.lognormal(0, 0.5, (n, 1)) for n in [400, 300, 500, 350]
    ]
    h = Households(days, np.array([15000, 15100, 15200, 15300]))
    rows, period_rows = h.offset[-1], h.period_offset[-1]
    entries = np.zeros((rows, 3, 2), np.float16)
    periods = np.zeros((period_rows, 3, 2), np.float16)
    for e, (d, p) in enumerate(zip(days, h.periods, strict=True)):
        entries[h.offset[e] : h.offset[e + 1], :, 0] = e
        entries[h.offset[e] : h.offset[e + 1], :, 1] = np.arange(len(d))[:, None]
        periods[h.period_offset[e] : h.period_offset[e + 1], :, 0] = e
        periods[h.period_offset[e] : h.period_offset[e + 1], :, 1] = -1 - np.arange(len(p))[:, None]
    return Split(
        h,
        entries,
        np.zeros((rows, N_STATS), np.float32),
        np.zeros(rows),
        periods,
        np.zeros((period_rows, N_STATS), np.float32),
        np.zeros(period_rows),
    )


def questions(sp: Split) -> list:
    rng = np.random.default_rng(1)
    return [ask(sp.h, f, rng, WINDOW) for f in FAMILIES[:5] for _ in range(4)]


def test_full_and_oracle_banks_hold_the_right_days_and_evidence_slots():
    sp = split()
    qs = questions(sp)
    for bank in ["oracle", "full"]:
        b_ = memory_for(sp, qs, bank, WINDOW, "cpu")
        ent, days, date, mask, slots = b_.vectors, b_.start, b_.date, b_.mask, b_.slots
        for b, q in enumerate(qs):
            want = q.evidence if bank == "oracle" else list(range(max(0, q.now - WINDOW), q.now))
            n = len(want)
            # The day entries come first, then the weeks and months.
            assert (b_.level[b][mask[b]] == 0).sum() == n and (b_.level[b, :n] == 0).all()
            assert torch.all(ent[b, :n, ::2] == q.household)
            assert ent[b, :n, 1].long().tolist() == want
            assert days[b, :n].long().tolist() == [q.now - j for j in want]
            assert (days[b, :n] - b_.end[b, :n] == 1).all()
            assert date[b, :n].tolist() == [sp.h.first[q.household] + j for j in want]
            on = slots[b].nonzero()[:, 0].tolist()
            if not q.evidence:
                assert on == [0]
            else:
                pos = [want.index(j) for j in q.evidence]
                assert on == [1 + p for p in pos]


def test_swap_and_none_banks_hold_nothing_from_the_household():
    sp = split()
    qs = questions(sp)
    b_ = memory_for(sp, qs, "swap", WINDOW, "cpu")
    ent, days, mask, slots = b_.vectors, b_.start, b_.mask, b_.slots
    for b, q in enumerate(qs):
        n = int(mask[b].sum())
        assert n > 0 and torch.all(ent[b, :n, ::2] == (q.household + 1) % 4)
        assert (days[b, :n] >= 1).all() and (days[b, :n] <= WINDOW).all()
        assert slots[b].nonzero()[:, 0].tolist() == ([0] if not q.evidence else [])
    assert not memory_for(sp, qs, "none", WINDOW, "cpu").mask.any()


def test_price_notices_join_the_bank_as_items_with_their_evidence():
    """Households 0 and 2 are on a schedule with a high price every 5th day and a low one every
    3rd."""
    sp = split()
    grid = np.zeros((300, 48), np.int8)
    grid[::5, 34:40] = 2
    grid[::3, 0:10] = 1
    sp.h.prices = Prices(grid, 15100, np.array([True, False, True, False]), 15100, 15400)
    sp.notices = torch.arange(300, dtype=torch.float16)[:, None].repeat(1, 4)
    rng = np.random.default_rng(3)
    qs = [ask(sp.h, "price_last", rng, WINDOW) for _ in range(6)]
    assert all(q.household in (0, 2) for q in qs)
    b = memory_for(sp, qs, "full", WINDOW, "cpu")
    notes, ago, date, mask = b.items, b.item_start, b.item_date, b.item_mask
    n = b.mask.shape[1]
    for i, q in enumerate(qs):
        m = int(mask[i].sum())
        d = date[i, :m].numpy()
        assert m > 0 and (sp.h.prices.grid[d - 15100].any(1)).all()
        assert notes[i, :m, 0].long().tolist() == (d - 15100).tolist()
        assert (ago[i, :m].long().numpy() == sp.h.first[q.household] + q.now - d).all()
        on = b.slots[i].nonzero()[:, 0].tolist()
        assert on == [1 + n + d.tolist().index(sp.h.first[q.household] + q.items[0])]


def test_weeks_and_months_join_the_entry_slots():
    sp = split()
    rng = np.random.default_rng(5)
    qs = [ask(sp.h, "month_change", rng, 400) for _ in range(8)]
    qs += [ask(sp.h, "recall", rng, 400) for _ in range(8)]
    b = memory_for(sp, qs, "full", 400, "cpu")
    ent, days, mask, level = b.vectors, b.start, b.mask, b.level
    for i, q in enumerate(qs):
        per = sp.h.periods[q.household]
        lo = max(0, q.now - 400)
        want = np.flatnonzero((per[:, 1] >= lo) & (per[:, 2] < q.now))
        n_days = q.now - lo
        k = int(mask[i].sum())
        assert k == n_days + len(want)
        assert (level[i, :n_days] == 0).all()
        assert level[i, n_days:k].tolist() == per[want, 0].tolist()
        assert ent[i, n_days:k, 1].long().tolist() == (-1 - want).tolist()
        assert days[i, n_days:k].long().tolist() == (q.now - per[want, 1]).tolist()
        assert b.end[i, n_days:k].long().tolist() == (q.now - per[want, 2] - 1).tolist()
        on = b.slots[i].nonzero()[:, 0].tolist()
        if q.family == "month_change":
            assert on == sorted(1 + n_days + want.tolist().index(r) for r in q.periods)
            assert all(per[r, 0] == 2 for r in q.periods)
    oracle = memory_for(sp, qs, "oracle", 400, "cpu")
    for i, q in enumerate(qs):
        assert int(oracle.mask[i].sum()) == len(q.evidence) + len(q.periods)
