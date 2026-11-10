"""Questions about a household's days, computed by code, each with the days its answer comes from.

The current day is in the TSLM's own input. Memory holds the days in [now - window, now)."""

from dataclasses import dataclass, field

import numpy as np

from tslm_memory.memory import calendar_levels

FAMILIES = [
    "recall",
    "last_time",
    "count",
    "baseline",
    "control",
    "price_last",
    "month_change",
    "month_total",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MIN_HISTORY = 56
# Thresholds keep every day at least 10% away from X, so reading a total roughly is enough.
MARGIN = 1.1
# Baseline: "about the same" within 15% of the usual day. Ratios within 5 points of that edge are
# skipped, since rounding error in reading either total would flip them.
SAME, GAP = np.log(1.15), 0.05
# Levels of the memory hierarchy. Week and month entries pool the day entries below them.
MONTH = 2  # the level of a month in calendar_levels


@dataclass
class Prices:
    """The time-of-use price schedule, the same for every household on it."""

    grid: np.ndarray  # [days, 48] 0 normal, 1 low, 2 high
    first: int  # date of the grid's first row
    tou: np.ndarray  # per household, on the schedule or not
    ask_from: int  # price questions only ask about events on dates in [ask_from, ask_to)
    ask_to: int

    def notice(self, date: int) -> str | None:
        """The text item of a date with low or high prices, None for a normal day."""
        r = date - self.first
        if not 0 <= r < len(self.grid) or not self.grid[r].any():
            return None
        row, parts, i = self.grid[r], [], 0
        while i < len(row):
            j = i
            while j < len(row) and row[j] == row[i]:
                j += 1
            if row[i]:
                name = "high" if row[i] == 2 else "low"
                parts.append(f"{name} price from {clock(i)} to {clock(j)}")
            i = j
        return "Price notice: " + ", ".join(parts) + "."


def clock(slot: int) -> str:
    return f"{slot // 2:02d}:{30 * (slot % 2):02d}"


@dataclass
class Households:
    days: list[np.ndarray]  # per household [n_days, 48] kWh, NaN for missing readings
    first: np.ndarray  # per household the date of its first day, days since 1970-01-01
    prices: Prices | None = None
    totals: list[np.ndarray] = field(init=False)  # per day, NaN unless all 48 readings are there
    offset: np.ndarray = field(init=False)  # [n + 1] row of each household's first day in a split
    periods: list[np.ndarray] = field(init=False)  # per household [P, 3] from calendar_levels()
    period_offset: np.ndarray = field(init=False)  # [n + 1] row of each household's first period
    reach: np.ndarray = field(init=False)  # [rows] per day, rows as in offset, from reach()

    def __post_init__(self):
        self.totals = [np.asarray(d, np.float64).sum(1) for d in self.days]
        self.offset = np.cumsum([0] + [len(d) for d in self.days])
        self.periods = [
            calendar_levels(int(f), len(d)) for f, d in zip(self.first, self.days, strict=True)
        ]
        self.period_offset = np.cumsum([0] + [len(p) for p in self.periods])
        self.reach = np.concatenate([reach(t) for t in self.totals])


def reach(t: np.ndarray) -> np.ndarray:
    """Per day j the longest distance d for which j can answer a last-time question asked d days
    later: some whole X has t[j] > MARGIN * X while every day between is below X / MARGIN, none
    missing. The largest such X is ceil(t[j] / MARGIN) - 1, so a later day ends the reach once it
    is at least that over MARGIN. 0 for missing days."""
    n = len(t)
    top = np.ceil(t / MARGIN) - 1
    later = np.nan_to_num(t, nan=np.inf)
    stop = (MARGIN * later[None, :] >= top[:, None]) & (
        np.arange(n)[None, :] > np.arange(n)[:, None]
    )
    first = np.where(stop.any(1), stop.argmax(1), n - 1)
    out = first - np.arange(n)
    out[np.isnan(t) | (top < 1)] = 0
    return out


def month_of(date: int | np.ndarray) -> int | np.ndarray:
    """Months since January 1970."""
    return np.asarray(date).astype("datetime64[D]").astype("datetime64[M]").astype(int)


@dataclass
class Question:
    household: int
    now: int  # day index of the current day
    family: str
    text: str
    answer: str
    evidence: list[int]  # day indices the answer comes from, empty for the controls
    items: list[int] = field(default_factory=list)  # day indices of text items it comes from
    periods: list[int] = field(default_factory=list)  # rows of h.periods[household] it comes from


def items(h: Households, e: int, lo: int, now: int) -> np.ndarray:
    """Day indices in [lo, now) that hold a price notice for household e."""
    p = h.prices
    if p is None or not p.tou[e]:
        return np.arange(0)
    j = np.arange(max(lo, p.first - h.first[e]), min(now, p.first + len(p.grid) - h.first[e]))
    return j[p.grid[h.first[e] + j - p.first].any(1)]


def weekday(date: int | np.ndarray) -> int | np.ndarray:
    return (date + 3) % 7  # 1970-01-01 was a Thursday


def kwh(x: float) -> int:
    return int(np.floor(x + 0.5))


def ask(h: Households, family: str, rng: np.random.Generator, window: int) -> Question:
    """Draw a question of one family, resampling until the household and day can answer it.

    For recall, last time and count the answer or distance is drawn first. Drawn per question,
    most days only give the easy answers (yesterday, none or all of the 14 days), and a
    distance drawn per day mostly falls in the first months. Distances are uniform over the
    window. A last-time answer d days back needs a day that beats every later day by 10%, which
    only about 0.05% of draws past 30 days find, so the household and day come from the days
    whose reach is at least d."""
    target, candidates = None, None
    for attempt in range(100_000):
        # A long answer can be missing from every household, so redraw it now and then.
        if attempt % 20_000 == 0:
            target = {
                "recall": int(rng.integers(1, window + 1)),
                "last_time": int(rng.integers(1, window + 1)),
                "count": int(rng.integers(0, 15)),
            }.get(family)
            candidates = np.flatnonzero(h.reach >= target) if family == "last_time" else None
        if candidates is not None:
            if not len(candidates):
                target = int(rng.integers(1, window + 1))
                candidates = np.flatnonzero(h.reach >= target)
                continue
            c = int(candidates[rng.integers(len(candidates))])
            e = int(np.searchsorted(h.offset, c, "right")) - 1
            now = c - int(h.offset[e]) + target
        else:
            e = int(rng.integers(len(h.days)))
            now = (
                int(rng.integers(MIN_HISTORY, len(h.totals[e])))
                if len(h.totals[e]) > MIN_HISTORY
                else 0
            )
        t = h.totals[e]
        if now < MIN_HISTORY or np.isnan(t[now]):
            continue
        q = generate(h, e, now, family, rng, window, target)
        if q is not None:
            return q
    raise RuntimeError(f"no {family} question with answer {target} in 100000 tries")


def generate(
    h: Households,
    e: int,
    now: int,
    family: str,
    rng: np.random.Generator,
    window: int,
    target: int | None = None,
) -> Question | None:
    t, lo = h.totals[e], max(0, now - window)
    if family == "recall":
        n = target
        if n > now - lo or np.isnan(t[now - n]):
            return None
        text = f"What was the total electricity use {n} days before today, in kWh?"
        return Question(e, now, family, text, str(kwh(t[now - n])), [now - n])
    if family == "last_time":
        j = now - target
        later = t[j + 1 : now]
        if j < lo or np.isnan(t[j]) or np.isnan(later).any():
            return None
        lowest = max(1, int(np.floor(later.max(initial=0) * MARGIN)) + 1)
        highest = int(np.ceil(t[j] / MARGIN)) - 1
        if highest < lowest:
            return None
        x = int(rng.integers(lowest, highest + 1))
        text = f"How many days before today did daily use last exceed {x} kWh?"
        return Question(e, now, family, text, str(target), [j])
    if family == "count":
        last = t[now - 14 : now]
        if now - 14 < lo or np.isnan(last).any():
            return None
        # Count k needs X between the k-th and (k + 1)-th largest totals, 10% clear of both.
        s = np.sort(last)[::-1]
        below = s[target] if target < 14 else 0.0
        above = s[target - 1] if target > 0 else 2 * s[0] + 2
        lowest = max(1, int(np.floor(below * MARGIN)) + 1)
        highest = int(np.ceil(above / MARGIN)) - 1
        if highest < lowest:
            return None
        x = int(rng.integers(lowest, highest + 1))
        text = f"On how many of the last 14 days did daily use exceed {x} kWh?"
        return Question(e, now, family, text, str(target), list(range(now - 14, now)))
    if family == "baseline":
        same = [now - 7 * k for k in range(1, 9)]
        if same[-1] < lo or np.isnan(t[same]).any() or t[now] <= 0:
            return None
        usual = np.median(t[same])
        if usual < 0.5:
            return None
        r = np.log(t[now] / usual)
        if abs(abs(r) - SAME) < GAP:
            return None
        answer = "about the same" if abs(r) < SAME else "higher" if r > 0 else "lower"
        name = WEEKDAYS[weekday(h.first[e] + now)]
        text = (
            f"Was today's total use higher, lower or about the same as on a usual {name} "
            "over the last 8 weeks?"
        )
        return Question(e, now, family, text, answer, sorted(same))
    if family == "price_last":
        p, j = h.prices, items(h, e, lo, now)
        high = j[(p.grid[h.first[e] + j - p.first] == 2).any(1)] if len(j) else j
        if not len(high) or not p.ask_from <= h.first[e] + high[-1] < p.ask_to:
            return None
        j = int(high[-1])
        text = "How many days before today was the last day with a high electricity price?"
        return Question(e, now, family, text, str(now - j), [], [j])
    if family in ("month_change", "month_total"):
        # The last complete calendar month, alone or against the same month a year before.
        p = h.periods[e]
        months = np.flatnonzero((p[:, 0] == MONTH) & (p[:, 2] < now) & (p[:, 1] >= lo))
        if not len(months) or p[months[-1], 2] < now - 31:
            return None
        last = months[-1]
        if family == "month_total":
            # To the nearest 10 kWh, skipped within 2 kWh of a rounding edge.
            a = t[p[last, 1] : p[last, 2] + 1].sum()
            if np.isnan(a) or abs(a / 10 - np.floor(a / 10) - 0.5) < 0.2:
                return None
            text = "What was the total electricity use last month, in kWh to the nearest 10?"
            return Question(
                e, now, family, text, str(10 * int(np.floor(a / 10 + 0.5))), [], [], [int(last)]
            )
        key = month_of(h.first[e] + p[months, 1])
        before = months[key == key[-1] - 12]
        if not len(before):
            return None
        a, b = (t[p[k, 1] : p[k, 2] + 1].sum() for k in (last, before[0]))
        if np.isnan(a) or np.isnan(b) or b <= 0 or a <= 0:
            return None
        r = np.log(a / b)
        if abs(abs(r) - SAME) < GAP:
            return None
        answer = "about the same" if abs(r) < SAME else "higher" if r > 0 else "lower"
        text = (
            "Was the total use last month higher, lower or about the same as in the same month "
            "a year before?"
        )
        return Question(e, now, family, text, answer, [], [], [int(last), int(before[0])])
    if family == "control":
        day = h.days[e][now]
        peak = int(np.argmax(day))
        others = np.delete(day, [2 * (peak // 2), 2 * (peak // 2) + 1])
        if day[peak] < MARGIN * others.max():
            return None
        text = "At which hour of the day, from 0 to 23, was today's use highest?"
        return Question(e, now, family, text, str(peak // 2), [])
    raise ValueError(family)
