import re

import numpy as np
import pytest

from tslm_memory.memory import calendar_levels
from tslm_memory.questions import (
    FAMILIES,
    MARGIN,
    Households,
    Prices,
    ask,
    generate,
    month_of,
    reach,
    weekday,
)

WINDOW = 100


def households(n: int = 6, seed: int = 0) -> Households:
    rng = np.random.default_rng(seed)
    days = []
    for _ in range(n):
        n_days = int(rng.integers(300, 600))
        d = (
            rng.gamma(2, 0.1, (n_days, 48))
            * rng.lognormal(0, 0.5, (n_days, 1))
            * rng.uniform(0.5, 3)
        )
        d[rng.random(n_days) < 0.01, rng.integers(48)] = np.nan
        days.append(d)
    return Households(days, rng.integers(15000, 16000, n))


def numbers(text: str) -> list[int]:
    # Hours like 06:00 count as one number.
    text = re.sub(r"(\d\d):00", r"\1", text)
    return [int(x) for x in re.findall(r"\d+", text)]


@pytest.mark.parametrize("family", [f for f in FAMILIES if not f.startswith(("price", "month"))])
def test_evidence_holds_the_answer(family: str):
    h = households()
    rng = np.random.default_rng(1)
    for _ in range(300):
        q = ask(h, family, rng, WINDOW)
        t = h.totals[q.household]
        ev = np.array(q.evidence, int)
        assert all(q.now - WINDOW <= j < q.now for j in q.evidence)
        assert not np.isnan(t[ev]).any() and not np.isnan(t[q.now])
        match family:
            case "recall":
                (n,) = numbers(q.text)
                assert q.evidence == [q.now - n]
                assert abs(t[ev[0]] - int(q.answer)) <= 0.5
            case "last_time":
                (x,) = numbers(q.text)
                assert q.evidence == [q.now - int(q.answer)]
                assert t[ev[0]] > x * MARGIN
                assert (t[ev[0] + 1 : q.now] < x / MARGIN).all()
            case "count":
                (_, x) = numbers(q.text)
                assert q.evidence == list(range(q.now - 14, q.now))
                assert int(q.answer) == (t[ev] > x).sum()
                assert not ((t[ev] > x / MARGIN) & (t[ev] < x * MARGIN)).any()
            case "baseline":
                first = h.first[q.household]
                assert len(ev) == 8 and (weekday(first + ev) == weekday(first + q.now)).all()
                r = t[q.now] / np.median(t[ev])
                want = "higher" if r > 1.15 else "lower" if r < 1 / 1.15 else "about the same"
                assert q.answer == want
            case "control":
                day = h.days[q.household][q.now]
                assert q.evidence == [] and int(q.answer) == np.argmax(day) // 2


def test_every_family_has_varied_answers():
    h = households()
    rng = np.random.default_rng(2)
    for family in [f for f in FAMILIES if not f.startswith(("price", "month"))]:
        answers = {ask(h, family, rng, WINDOW).answer for _ in range(200)}
        assert len(answers) >= 3, family


def test_price_questions_ask_about_the_last_high_day_in_the_allowed_dates():
    h = households()
    grid = np.zeros((365, 48), np.int8)
    grid[::4, 34:40] = 2
    grid[1::7, 0:10] = 1
    tou = np.array([True, False] * 3)
    lo = int(h.first.min()) + 100
    h.prices = Prices(grid, lo, tou, lo + 200, lo + 365)
    rng = np.random.default_rng(4)
    for _ in range(100):
        q = ask(h, "price_last", rng, WINDOW)
        e, (j,) = q.household, q.items
        assert tou[e] and lo + 200 <= h.first[e] + j < lo + 365
        later = h.first[e] + np.arange(j + 1, q.now) - lo
        later = later[(later >= 0) & (later < 365)]
        assert grid[h.first[e] + j - lo].max() == 2 and grid[later].max(initial=0) < 2
        assert q.evidence == [] and int(q.answer) == q.now - j
    assert h.prices.notice(lo + 1) == "Price notice: low price from 00:00 to 05:00."
    assert h.prices.notice(lo) == "Price notice: high price from 17:00 to 20:00."
    assert h.prices.notice(lo + 2) is None


def test_month_change_compares_last_month_with_the_same_month_a_year_before():
    h = households(n=6, seed=3)
    rng = np.random.default_rng(6)
    for _ in range(50):
        q = ask(h, "month_change", rng, 600)
        per, t = h.periods[q.household], h.totals[q.household]
        last, before = (per[r] for r in q.periods)
        assert last[0] == before[0] == 2 and last[2] < q.now <= last[2] + 31
        first = h.first[q.household]
        assert month_of(first + last[1]) - month_of(first + before[1]) == 12
        r = t[last[1] : last[2] + 1].sum() / t[before[1] : before[2] + 1].sum()
        want = "higher" if r > 1.15 else "lower" if r < 1 / 1.15 else "about the same"
        assert q.answer == want and q.evidence == []


def test_periods_are_complete_calendar_weeks_and_months():
    first = 15339  # 2011-12-31, a Saturday
    p = calendar_levels(first, 100)
    weeks, months = p[p[:, 0] == 1], p[p[:, 0] == 2]
    assert (weekday(first + weeks[:, 1]) == 0).all() and (weeks[:, 2] - weeks[:, 1] == 6).all()
    assert months[:, 1].tolist() == [1, 32, 61] and months[:, 2].tolist() == [31, 60, 91]


def test_month_total_reads_last_month_to_the_nearest_10():
    h = households(n=6, seed=3)
    rng = np.random.default_rng(8)
    for _ in range(50):
        q = ask(h, "month_total", rng, 600)
        (r,) = q.periods
        level, a, b = h.periods[q.household][r]
        assert level == 2 and b < q.now <= b + 31
        assert abs(h.totals[q.household][a : b + 1].sum() - int(q.answer)) <= 5 - 2


def test_reach_is_the_longest_distance_with_a_last_time_question():
    h = households(n=2, seed=5)
    rng = np.random.default_rng(0)
    t = h.totals[0]
    r = reach(t)
    for j in rng.integers(0, len(t) - 2, 200):
        for d in [int(r[j]), int(r[j]) + 1]:
            now = j + d
            if d == 0 or now >= len(t) or np.isnan(t[now]):
                continue
            q = generate(h, 0, now, "last_time", rng, 10_000, d)
            assert (q is not None) == (d <= r[j]), (j, d, r[j])


def test_recall_and_last_time_distances_spread_over_the_window():
    h = households(n=6, seed=1)
    rng = np.random.default_rng(3)
    d = np.array([numbers(ask(h, "recall", rng, 300).text)[0] for _ in range(300)])
    assert (d > 150).mean() > 0.3
    # Last time spreads up to the longest distance any day can answer.
    top = min(300, h.reach.max())
    d = np.array([int(ask(h, "last_time", rng, 300).answer) for _ in range(300)])
    assert d.max() <= top and (d > top / 2).mean() > 0.3, top
