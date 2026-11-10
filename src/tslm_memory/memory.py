"""The memory: entries built from what a dataset supplies, and blocks that read them from inside a
decoder LLM. One code path for every dataset.

A dataset supplies base entries (encoder vectors, statistics computed by code, each entry's
start and end on its own clock) and text items. Coarser levels form a pyramid over the base
entries, by calendar units or by a base unit times a factor. A coarser entry is pooled from its
base entries: the mean of their vectors, and statistics recomputed from theirs. Every entry and
every item is one memory slot holding its vectors or text, statistics and span. Compression, the
comparison, replaces the slots with the mean slot of each of n equal time chunks."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Settings:
    """What a dataset sets. Spans are in the clock's unit from the clock's reference point, which
    the dataset's bank builder applies (London: days before now; TS-Haystack: seconds from the
    window start)."""

    levels: int  # pyramid levels, the base first
    periods: tuple[float, float, int]  # linear time features: shortest and longest period, count
    calendar: bool  # spans count days, and each entry's date gives weekday and day of year
    exact: int = 0  # statistics given as exact values, as Fourier features on a linear scale
    exact_periods: tuple[float, float, int] = (1, 2048, 12)


def calendar_levels(first: int, n: int) -> np.ndarray:
    """[P, 3] level (1 week, 2 month), first and last base index of the complete calendar weeks
    (Monday to Sunday) and months among days [0, n) whose day 0 is on date first, weeks first."""
    dates = first + np.arange(n)
    month = dates.astype("datetime64[D]").astype("datetime64[M]").astype(int)
    out = []
    for level, key in [(1, (dates + 3) // 7), (2, month)]:
        _, start, count = np.unique(key, return_index=True, return_counts=True)
        if level == 1:
            size = np.full(len(start), 7)
        else:
            m = np.datetime64(0, "M") + month[start].astype("timedelta64[M]")
            end = m + np.timedelta64(1, "M")
            size = (end.astype("datetime64[D]") - m.astype("datetime64[D]")).astype(int)
        done = count == size
        out += [(level, a, a + c - 1) for a, c in zip(start[done], count[done], strict=True)]
    return np.array(out, int).reshape(-1, 3)


def grid_levels(n: int, factor: int, levels: int) -> np.ndarray:
    """[P, 3] level, first and last base index of the complete groups of factor**level base
    entries among [0, n), aligned at base entry 0, for levels 1 to levels - 1."""
    out = [np.zeros((0, 3), int)]
    for level in range(1, levels):
        size = factor**level
        a = np.arange(n // size) * size
        out.append(np.stack([np.full(len(a), level), a, a + size - 1], 1))
    return np.concatenate(out)


def pool_vectors(x: torch.Tensor, groups: np.ndarray) -> torch.Tensor:
    """The mean of base vectors x [n, ...] over each group, [P, ...] float32."""
    cs = torch.cat([torch.zeros_like(x[:1], dtype=torch.float32), x.float().cumsum(0)])
    a = torch.from_numpy(groups[:, 1]).to(x.device)
    b = torch.from_numpy(groups[:, 2] + 1).to(x.device)
    return (cs[b] - cs[a]) / (b - a).view(-1, *[1] * (x.dim() - 1))


def pool_stats(raw: np.ndarray, rules: list, groups: np.ndarray) -> np.ndarray:
    """Base statistics raw [n, S] over each group, [P, S], one rule per column: "sum", "min",
    "max", or a column index k for the value at the base entry where column k is largest."""
    a, b = groups[:, 1], groups[:, 2] + 1
    out = np.zeros((len(groups), raw.shape[1]), np.float32)
    if not len(groups):
        return out
    # reduceat needs every index below the length, and b reaches n.
    pad = np.concatenate([raw, raw[-1:]])
    idx = np.stack([a, b], 1).ravel()
    for k, rule in enumerate(rules):
        if rule == "sum":
            cs = np.concatenate([[0.0], np.cumsum(raw[:, k], dtype=np.float64)])
            out[:, k] = cs[b] - cs[a]
        elif rule in ("min", "max"):
            out[:, k] = (np.minimum if rule == "min" else np.maximum).reduceat(pad[:, k], idx)[::2]
        else:
            out[:, k] = [raw[x + np.argmax(raw[x:y, rule]), k] for x, y in zip(a, b, strict=True)]
    return out


@dataclass
class Part:
    """One sample's entries or text items before padding."""

    vectors: torch.Tensor  # [n, F] encoder vectors, or [n, hidden] text vectors for items
    start: np.ndarray  # [n] on the dataset's clock
    end: np.ndarray
    evidence: np.ndarray  # [n] bool, the slots that answer the question
    stats: np.ndarray | None = None  # [n, S] features, entries only
    exact: np.ndarray | None = None  # [n, E] exact values, entries only
    level: np.ndarray | None = None  # [n] pyramid level, entries only
    date: np.ndarray | None = None  # [n] days since 1970, with a calendar
    names: list[str] | None = None  # per slot, as a recall writes it
    label: np.ndarray | None = None  # [n] class of what an entry holds, -1 for none


@dataclass
class Bank:
    """A padded batch: entries [B, N, ...], then text items [B, M, ...]. slots [B, 1 + N + M] is
    true at the evidence, slot 0 the null entry."""

    vectors: torch.Tensor
    stats: torch.Tensor
    exact: torch.Tensor
    level: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    date: torch.Tensor
    mask: torch.Tensor
    items: torch.Tensor
    item_start: torch.Tensor
    item_end: torch.Tensor
    item_date: torch.Tensor
    item_mask: torch.Tensor
    slots: torch.Tensor
    names: list[list[str]] = field(default_factory=list)  # per sample, slots 1 to N + M
    label: torch.Tensor | None = None  # [B, N] entry classes, -1 for none or padding


def batch(
    entries: list[Part],
    items: list[Part],
    null: list[bool],
    dims: tuple[int, int, int, int],
    device: str,
) -> Bank:
    """Pads one Part of entries and one of items per sample. null marks the samples whose
    evidence is the null entry. dims: vector width, statistics, exact values, text width."""
    b = len(entries)
    n = max(len(p.start) for p in entries)
    m = max(len(p.start) for p in items)
    f, s, e, h = dims

    def zeros(*shape, dtype=torch.float32):
        return torch.zeros(b, *shape, dtype=dtype, device=device)

    vectors, stats, exact = zeros(n, f, dtype=torch.float16), zeros(n, s), zeros(n, e)
    level, date, item_date = (zeros(k, dtype=torch.long) for k in (n, n, m))
    start, end, item_start, item_end = zeros(n), zeros(n), zeros(m), zeros(m)
    mask, item_mask = zeros(n, dtype=torch.bool), zeros(m, dtype=torch.bool)
    texts = zeros(m, h, dtype=torch.float16)
    slots = zeros(1 + n + m, dtype=torch.bool)
    label = torch.full((b, n), -1, dtype=torch.long, device=device)
    names = []
    for i, (p, q) in enumerate(zip(entries, items, strict=True)):
        k, j = len(p.start), len(q.start)
        vectors[i, :k] = p.vectors.to(device)
        if p.stats is not None:
            stats[i, :k] = torch.from_numpy(p.stats)
        if p.exact is not None:
            exact[i, :k] = torch.from_numpy(p.exact)
        if p.level is not None:
            level[i, :k] = torch.from_numpy(p.level)
        if p.date is not None:
            date[i, :k] = torch.from_numpy(p.date)
        if p.label is not None:
            label[i, :k] = torch.from_numpy(p.label)
        if q.date is not None:
            item_date[i, :j] = torch.from_numpy(q.date)
        start[i, :k], end[i, :k] = torch.from_numpy(p.start), torch.from_numpy(p.end)
        item_start[i, :j], item_end[i, :j] = torch.from_numpy(q.start), torch.from_numpy(q.end)
        mask[i, :k], item_mask[i, :j] = True, True
        texts[i, :j] = q.vectors.to(device)
        slots[i, 0] = null[i]
        slots[i, 1 : 1 + k] = torch.from_numpy(p.evidence)
        slots[i, 1 + n : 1 + n + j] = torch.from_numpy(q.evidence)
        pn, qn = p.names or [""] * k, q.names or [""] * j
        names.append(pn + [""] * (n - k) + qn + [""] * (m - j))
    return Bank(
        vectors, stats, exact, level, start, end, date, mask,
        texts, item_start, item_end, item_date, item_mask, slots, names, label,
    )  # fmt: skip


class Span(nn.Module):
    """Learned embedding of an entry's span on the dataset's clock: Fourier features of
    log(1 + t) and of t on a linear scale for its start and end, the log of its duration, and
    with a calendar the weekday and day of year of its date."""

    def __init__(self, dim: int, settings: Settings, n_log: int = 32, max_freq: float = 200.0):
        super().__init__()
        # Up to 200 rad per unit of log time, so neighbouring days stay apart at 150 days back.
        self.register_buffer("freqs", torch.logspace(-1, math.log10(max_freq), n_log))
        # Log time names a day only to 1% or so past 200 days back. The linear periods name it
        # exactly at any distance.
        lo, hi, n = settings.periods
        periods = torch.logspace(math.log2(lo), math.log2(hi), n, base=2)
        self.register_buffer("linear", 2 * math.pi / periods)
        self.calendar = settings.calendar
        width = 4 * n_log + 4 * n + 1 + (11 if self.calendar else 0)
        self.mlp = nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Linear(512, dim))

    def forward(self, start: torch.Tensor, end: torch.Tensor, date: torch.Tensor) -> torch.Tensor:
        t = torch.stack([start, end], -1).float()
        x = torch.log1p(t.clamp(min=0))[..., None] * self.freqs
        lin = t[..., None] * self.linear
        dur = torch.log1p((end - start).abs().float())[..., None]
        parts = [x.sin(), x.cos(), lin.sin(), lin.cos()]
        parts = [p.flatten(-2) for p in parts] + [dur]
        if self.calendar:
            # 1970-01-01 was a Thursday, so (date + 3) % 7 is 0 on Mondays.
            parts.append(nn.functional.one_hot((date + 3) % 7, 7).float())
            year = 2 * math.pi * (date % 365.25)[..., None] / 365.25
            year = year * torch.arange(1, 3, device=t.device)
            parts += [year.sin(), year.cos()]
        return self.mlp(torch.cat(parts, -1))


class MemoryBlock(nn.Module):
    """Gated cross-attention plus feed-forward, as in Flamingo, with both gates starting at 0.
    Both add only at positions from start onward."""

    def __init__(self, hidden: int, heads: int, head_dim: int):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.norm = nn.LayerNorm(hidden)
        self.mem_norm = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, heads * head_dim, bias=False)
        self.kv = nn.Linear(hidden, 2 * heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, hidden, bias=False)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))
        # [B, heads, T, N], for the retrieval loss and diagnostics
        self.last_attn: torch.Tensor | None = None

    def forward(
        self, h: torch.Tensor, mem: torch.Tensor, mask: torch.Tensor, start: torch.Tensor | None
    ) -> torch.Tensor:
        x = h.float()
        b, t, _ = x.shape
        q = self.q(self.norm(x)).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(self.mem_norm(mem)).view(b, -1, 2, self.heads, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        attn = scores.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1)
        self.last_attn = attn
        a = (attn @ v).transpose(1, 2).reshape(b, t, -1)
        on = 1.0
        if start is not None:
            on = (torch.arange(t, device=x.device)[None, :] >= start[:, None]).float()[..., None]
        x = x + on * torch.tanh(self.attn_gate) * self.o(a)
        x = x + on * torch.tanh(self.ff_gate) * self.ff(x)
        return x.to(h.dtype)


def time_chunks(mask: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, n: int) -> torch.Tensor:
    """[B, n, S] true where chunk i holds slot j: slot 0 (null) always, a real slot when the
    midpoint of its span [lo, hi] lies in the i-th of n equal parts of the sample's time range."""
    real = mask.clone()
    real[:, 0] = False
    first = torch.where(real, lo, torch.inf).amin(1, keepdim=True)
    last = torch.where(real, hi, -torch.inf).amax(1, keepdim=True)
    mid = (lo + hi) / 2
    at = ((mid - first) / (last - first).clamp(min=1e-6) * n).nan_to_num(0).floor()
    at = at.clamp(0, n - 1).long()
    allow = (at[:, None, :] == torch.arange(n, device=mask.device)[None, :, None]) & real[:, None]
    allow[:, :, 0] = True
    return allow


class Memory(nn.Module):
    """Entry and item projections, span embedding, null entry and blocks. Load a bank, then run
    the LLM."""

    def __init__(
        self,
        settings: Settings,
        entry_dim: int,  # all encoder vectors of an entry, concatenated
        n_stats: int,
        hidden: int,
        n_blocks: int,
        heads: int = 16,
        head_dim: int = 64,
        compress: int = 0,  # time chunks whose mean slots replace the slots for the blocks
    ):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(entry_dim, hidden), nn.LayerNorm(hidden))
        # One statistics projection per pyramid level. Through London's day projection a month's
        # total, on a scale it never saw, was read only to a median 12%.
        self.stats_proj = nn.ModuleList(
            nn.Sequential(
                nn.Linear(n_stats, 512), nn.GELU(), nn.Linear(512, hidden), nn.LayerNorm(hidden)
            )
            for _ in range(settings.levels)
        )
        # Exact values as Fourier features on a linear scale (London: the whole total, periods
        # 1 to 2048 kWh, so an entry names its total to the kWh), into the first layer of each
        # level's projection and zero at the start.
        lo, hi, n = settings.exact_periods
        periods = torch.logspace(math.log2(lo), math.log2(hi), n, base=2)
        self.register_buffer("exact_freqs", 2 * math.pi / periods)
        self.exact_in = nn.ModuleList(
            nn.Linear(2 * n * settings.exact, 512, bias=False) for _ in range(settings.levels)
        )
        for lin in self.exact_in:
            nn.init.zeros_(lin.weight)
        # Text items arrive as the frozen LLM's mean hidden state over their tokens.
        self.text_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.span = Span(hidden, settings)
        # The null entry gives attention somewhere to go when no memory is needed.
        self.null = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.blocks = nn.ModuleList(MemoryBlock(hidden, heads, head_dim) for _ in range(n_blocks))
        self.compress = compress
        self.mem: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        # [B, N] what each entry holds before its span is added, for a perception loss.
        self.content: torch.Tensor | None = None
        # [B, n, S] with mean pooling, which slots each chunk token holds.
        self.chunk_of: torch.Tensor | None = None
        # [B] the first position the blocks act on, the question's first token; None for all.
        # The blocks then leave the TSLM's own series tokens and the text before them as they are.
        self.start: torch.Tensor | None = None

    def stats(self, bank: Bank) -> torch.Tensor:
        """Each entry's statistics through its own level's projection."""
        f = bank.exact.clamp(-1e6, 1e6)[..., None] * self.exact_freqs
        exact = torch.cat([f.sin(), f.cos()], -1).flatten(-2)
        out = torch.zeros(*bank.level.shape, self.null.shape[-1], device=bank.level.device)
        for k, (p, e) in enumerate(zip(self.stats_proj, self.exact_in, strict=True)):
            sel = bank.level == k
            if sel.any():
                h = p[0](bank.stats[sel])
                if e.in_features:
                    h = h + e(exact[sel])
                out[sel] = p[3](p[2](p[1](h)))
        return out

    def load(self, bank: Bank) -> None:
        """Each entry is one slot holding its encoder vectors, statistics and span. With the
        statistics in a slot of their own, retrieval in London stage 1 settled on the encoder
        slots of the right day and never read the level from the statistics slot. Each text item
        is one slot too. The memory is [B, 1 + N + M], with the null entry first, or with
        compression [B, 1 + n], the null entry and each time chunk's mean slot."""
        b = bank.mask.shape[0]
        self.content = self.proj(bank.vectors.float()) + self.stats(bank)
        x = self.content + self.span(bank.start, bank.end, bank.date)
        y = self.text_proj(bank.items.float()) + self.span(
            bank.item_start, bank.item_end, bank.item_date
        )
        self.mem = torch.cat([self.null.expand(b, 1, -1), x, y], 1)
        null = torch.ones(b, 1, dtype=torch.bool, device=bank.mask.device)
        self.mask = torch.cat([null, bank.mask, bank.item_mask], 1)
        if not self.compress:
            return
        # Each slot's span as [lo, hi] on the clock, whichever way the clock runs.
        zero = torch.zeros(b, 1, device=bank.mask.device)
        a = torch.cat([zero, bank.start, bank.item_start], 1)
        e = torch.cat([zero, bank.end, bank.item_end], 1)
        lo, hi = torch.minimum(a, e), torch.maximum(a, e)
        # The mean slot of each chunk, with nothing learned, after the null entry. Empty chunks
        # are masked out. Learned resamplers trained without a retrieval loss read nothing in our
        # budget (tag stage-4); the chunk means take the same retrieval loss as the slots.
        inside = time_chunks(self.mask, lo, hi, self.compress)
        inside[:, :, 0] = False
        self.chunk_of = inside
        w = inside.float()
        means = (w @ self.mem) / w.sum(-1, keepdim=True).clamp(min=1)
        self.mem = torch.cat([self.mem[:, :1], means], 1)
        self.mask = torch.cat([self.mask[:, :1], inside.any(-1)], 1)

    def evidence(self, slots: torch.Tensor) -> torch.Tensor:
        """The evidence [B, S] of the bank's slots on the tokens the blocks read: the same slots,
        or with mean pooling the chunks that hold an evidence slot and the null when none does."""
        if self.chunk_of is None:
            return slots
        hit = (self.chunk_of & slots[:, None, :]).any(-1)
        return torch.cat([~hit.any(-1, keepdim=True), hit], 1)

    def attach(self, layers: nn.ModuleList, every: int = 4) -> list[int]:
        """Run a block before every `every`-th decoder layer. Returns the layer indices."""
        at = [i for i in range(len(layers)) if (i + 1) % every == 0]
        assert len(at) == len(self.blocks)
        for i, block in zip(at, self.blocks, strict=True):
            layers[i].register_forward_pre_hook(self._hook(block), with_kwargs=True)
        return at

    def _hook(self, block: MemoryBlock):
        def hook(module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            if args:
                return (block(args[0], self.mem, self.mask, self.start), *args[1:]), kwargs
            h = kwargs["hidden_states"]
            kwargs["hidden_states"] = block(h, self.mem, self.mask, self.start)
            return args, kwargs

        return hook
