"""Train the memory on TS-Haystack, LTAF ECG.

Each question is about one window of a recording, 15 min to 2 h long. The window is held in
memory and the TSLM's own input is the question. Banks:
  events: the window's annotated events as text items, rhythm bouts and beats that are not N,
      each with its kind, ordinal, start and end. The upper bound, like the paper's real oracle.
  tiles: the window's 10 s base entries, each the mean of four 2.5 s frozen encoder vectors
      with statistics computed by code pooled from theirs (ltaf_tiles), and coarser entries of
      40 s, 160 s and 640 s pooled the same way.
  swap: the tiles of another record of the split at the same offset.
  none: only the null entry, the no-memory baseline.
Training is on the bank given by --bank. With --compress the blocks read the mean slot of each
time chunk of the bank instead of its entries (memory.py), and --perceive adds a loss that
predicts each train 10 s entry's rhythm from what it holds. Answers are generated greedily and
scored with the benchmark's own rules (haystack_score)."""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from tslm_memory.haystack_score import correct
from tslm_memory.ltaf import HZ, RHYTHMS, ROOT, TASKS, clock, events, meta, questions
from tslm_memory.ltaf_tiles import TILE
from tslm_memory.memory import (
    Bank,
    Memory,
    Part,
    Settings,
    batch,
    grid_levels,
    pool_stats,
    pool_vectors,
)
from tslm_memory.tslm import add_lora, answer_logits, load_host, loss_and_exact

OUT = Path("runs/haystack")
GRID = 5 * HZ  # every window starts on a 5 s grid
FACTOR = 4  # 10 s, 40 s, 160 s, 640 s, each level of four entries of the one below
# A clock of seconds from the window start without a calendar, linear periods 1 s to 8192 s so
# an entry names its second anywhere in 2 h, and four pyramid levels.
HAYSTACK = Settings(levels=4, periods=(1, 8192, 14), calendar=False)
# How ltaf_tiles' statistics pool: per lead the sums, sums of squares, minima and maxima, then
# the R-peak count and the sums of RR intervals and their squares.
RULES = ["sum"] * 4 + ["min"] * 2 + ["max"] * 2 + ["sum"] * 3
N_STATS = 11
NAMES = {
    "NSR": "normal sinus rhythm",
    "AFIB": "atrial fibrillation",
    "SBR": "sinus bradycardia",
    "AB": "atrial bigeminy",
    "SVTA": "supraventricular tachyarrhythmia",
    "B": "ventricular bigeminy",
    "VT": "ventricular tachycardia",
    "T": "ventricular trigeminy",
    "IVR": "idioventricular rhythm",
    "A": "atrial premature beat (APC)",
    "V": "premature ventricular contraction (PVC)",
    "Q": "unclassifiable beat",
}


def item_text(kind: str, label: str, ordinal: int) -> str:
    noun = "bout" if kind == "bout" else "beat"
    return f"{label}, {NAMES[label]}: {noun} number {ordinal} of {label} in the recording"


def evidence(q: pd.Series) -> list[tuple[str, int, int]]:
    """Spans of the window, in samples from its start, that answer the question: ("bout", a, b)
    for rhythm bouts or gaps, ("beat", p, p) for single beats. Empty for a negative answer."""
    m, task = meta(q), q.task_type
    if task in ("localization", "comparison"):
        return [("bout", m["start_sample"], m["end_sample"])]
    if task in ("antecedent", "multi_hop"):
        return [
            ("bout", m["start_sample"], m["end_sample"]),
            ("bout", m["context_start_sample"], m["context_end_sample"]),
        ]
    if task == "ordering":
        return [("bout", m["a_start"], m["a_end"]), ("bout", m["b_start"], m["b_end"])]
    if task == "counting":
        return [("bout", a, b) for a, b in m["bout_segments"]]
    if task == "existence":
        if not m["is_positive"]:
            return []
        bouts, _ = events(q.record_id, q.start, q.context_length_samples)
        b = bouts[bouts.label == m["target_activity"]]
        return [("bout", a, e) for a, e in zip(b.start, b.end, strict=True)]
    if task == "anomaly_detection":
        return [("beat", p, p) for p in m["anomaly_beat_samples"]]
    if task == "state_query":
        return [("beat", m["beat_sample"], m["beat_sample"]), ("at", m["beat_sample"], 0)]
    return [("beat", m["beat_sample"], m["beat_sample"])]  # anomaly_localization


def features(raw: np.ndarray, n_base: np.ndarray) -> np.ndarray:
    """Pooled raw statistics [P, 11] of entries of n_base base entries -> per lead mean, std,
    min and max in mV, beats per second, mean RR and its std in s. Clipped, since electrode
    noise gives tens of mV."""
    n = (n_base * TILE)[:, None].astype(np.float32)
    mean = raw[:, 0:2] / n
    std = np.sqrt(np.maximum(raw[:, 2:4] / n - mean**2, 0))
    beats = raw[:, 8:9]
    k = np.maximum(beats, 1)
    rr = raw[:, 9:10] / k
    rr_std = np.sqrt(np.maximum(raw[:, 10:11] / k - rr**2, 0))
    out = [mean, std, raw[:, 4:8], beats / (n / HZ), rr, rr_std]
    return np.clip(np.concatenate(out, 1), -10, 10).astype(np.float32)


class Base:
    """ltaf_tiles' base entries of every record: vectors [rows, V, d] fp16 and raw statistics
    [rows, 11], memory-mapped. Row j of a record covers samples [j * TILE, (j + 1) * TILE)."""

    def __init__(self):
        self.x = np.load(ROOT / "base_chronos.npy", mmap_mode="r")
        self.raw = np.load(ROOT / "base_chronos_stats.npy", mmap_mode="r")
        index = np.load(ROOT / "base_chronos_index.npz")
        self.offset = dict(zip(index["records"], index["offset"], strict=True))
        self.count = dict(zip(index["records"], index["count"], strict=True))

    def rows(self, record: str, start: int, n: int) -> slice:
        assert start % TILE == 0 and n % TILE == 0
        a = self.offset[record] + start // TILE
        return slice(a, a + n // TILE)


def hits(a: np.ndarray, e: np.ndarray, level: np.ndarray, ev: list, n: int) -> np.ndarray:
    """Pyramid entries spanning samples [a, e) that are evidence: a base entry that overlaps an
    evidence span, a coarser one with at least half of its span inside evidence spans. Counting
    every coarse entry that touches the evidence let the 640 s entries satisfy the retrieval
    loss alone, and recall then wrote 640 s spans."""
    edges = np.zeros(n + 2, int)
    for _, lo, hi in ev:
        lo = min(max(lo, 0), n)
        edges[lo] += 1
        edges[min(max(hi, lo + 1), n + 1)] -= 1
    inside = np.concatenate([[0], np.cumsum(np.cumsum(edges)[: n + 1] > 0)])
    covered = inside[np.minimum(e, n + 1)] - inside[a]
    return np.where(level == 0, covered > 0, 2 * covered >= e - a)


class Data:
    """A split's questions with their windows' events, and the banks for a batch."""

    def __init__(self, split: str, lengths: list[int], base: Base | None):
        self.q = questions(split, lengths)
        self.base = base
        self.perceive = False  # label each own 10 s entry with its rhythm, train split only
        self.dim = base.x.shape[1] * base.x.shape[2] if base else 1
        self.hidden = 1  # width of the text item vectors, set once the host is loaded
        self.records = sorted(self.q.record_id.unique())
        windows = zip(self.q.record_id, self.q.start, self.q.context_length_samples, strict=True)
        self.events = {w: events(*w) for w in set(windows)}
        self.ev = [evidence(q) for q in self.q.itertuples()]

    def swapped(self, record: str, start: int, n: int) -> tuple[str, int]:
        """The next record of the split, at the same offset or as late as it allows."""
        other = self.records[(self.records.index(record) + 1) % len(self.records)]
        last = self.base.count[other] * TILE - n
        return other, min(start, last // GRID * GRID)

    def tiles(self, record: str, start: int, n: int, ev: list, device: str) -> Part:
        """The window's 10 s entries and the complete coarser groups of FACTOR**level of them,
        aligned at the window start, all pooled from the 2.5 s rows. A
        linear rhythm probe reads one 10 s entry pooled this way at macro-F1 0.638 against 0.578
        for one 2.5 s row."""
        sl = self.base.rows(record, start, n)
        x = torch.from_numpy(np.asarray(self.base.x[sl])).to(device).flatten(1)
        raw = np.asarray(self.base.raw[sl])
        # Groups of 4**(level + 1) rows of 2.5 s, so level 0 is 10 s.
        groups = grid_levels(len(x), FACTOR, HAYSTACK.levels + 1)
        groups[:, 0] -= 1
        vectors = pool_vectors(x, groups).half()
        st = features(pool_stats(raw, RULES, groups), groups[:, 2] - groups[:, 1] + 1)
        a, e = groups[:, 1] * TILE, (groups[:, 2] + 1) * TILE
        return Part(
            vectors,
            (a / HZ).astype(np.float32),
            (e / HZ).astype(np.float32),
            hits(a, e, groups[:, 0], ev, n),
            stats=st,
            exact=np.zeros((len(a), 0), np.float32),
            level=groups[:, 0],
        )

    def rhythm(self, record: str, start: int, n: int, p: Part) -> np.ndarray:
        """Per entry the index in RHYTHMS of the annotated rhythm at the centre of each 10 s entry,
        -1 for coarser entries and where no rhythm is annotated."""
        bouts = self.events[record, start, n][0].sort_values("start")
        c = (p.start + p.end) / 2 * HZ
        j = np.searchsorted(bouts.start.to_numpy(), c, side="right") - 1
        ok = (j >= 0) & (p.level == 0)
        j = np.maximum(j, 0)
        ok &= c < bouts.end.to_numpy()[j] if len(bouts) else False
        code = np.array([RHYTHMS.index(x) for x in bouts.label]) if len(bouts) else np.zeros(1)
        return np.where(ok, code[j], -1).astype(np.int64)

    def items(self, i: int, texts: dict, vectors: torch.Tensor) -> Part:
        """One question's annotated events as text items."""
        q, ev = self.q.iloc[i], self.ev[i]
        r, s, n = q.record_id, q.start, q.context_length_samples
        # A count is the ordinal of the last bout of its rhythm, so that one bout answers.
        ev = ev[-1:] if q.task_type == "counting" else ev
        bouts, beats = self.events[r, s, n]
        ids = [texts["bout", lab, o] for lab, o in zip(bouts.label, bouts.ordinal, strict=True)]
        ids += [texts["beat", lab, o] for lab, o in zip(beats.label, beats.ordinal, strict=True)]
        a = np.concatenate([bouts.start, beats.start]).astype(np.float32)
        e = np.concatenate([bouts.end, beats.start]).astype(np.float32)
        beat = np.arange(len(a)) >= len(bouts)
        hit = np.zeros(len(a), bool)
        for k, lo, hi in ev:
            if k == "bout":
                hit |= ~beat & (a < max(hi, lo + 1)) & (e > lo)
            elif k == "beat":
                hit |= beat & (a == lo)
            else:  # the rhythm at a beat
                hit |= ~beat & (a <= lo) & (e > lo)
        x = vectors[ids] if ids else torch.zeros(0, self.hidden)
        names = list(bouts.label) + list(beats.label)
        return Part(x, a / HZ, e / HZ, hit, names=names)

    def bank(
        self, idx: np.ndarray, kind: str, texts: dict, vectors: torch.Tensor | None, device: str
    ) -> Bank:
        empty = np.zeros(0, np.float32)
        entries, items, null = [], [], []
        for i in idx:
            q = self.q.iloc[i]
            r, s, n = q.record_id, q.start, q.context_length_samples
            e = Part(torch.zeros(0, self.dim), empty, empty, np.zeros(0, bool))
            t = Part(torch.zeros(0, self.hidden), empty, empty, np.zeros(0, bool))
            if kind == "tiles":
                e = self.tiles(r, s, n, self.ev[i], device)
                e.label = self.rhythm(r, s, n, e) if self.perceive else None
            elif kind == "swap":
                e = self.tiles(*self.swapped(r, s, n), n, [], device)
            elif kind == "events":
                t = self.items(i, texts, vectors)
            entries.append(e)
            items.append(t)
            null.append(not (e.evidence.any() or t.evidence.any()))
        return batch(entries, items, null, (self.dim, N_STATS, 0, self.hidden), device)


def duration(n: int) -> str:
    s = n // HZ
    return f"{s // 60} minutes" if s < 3600 else f"{s // 3600} hour" + ("s" if s >= 7200 else "")


def prompts(rows: pd.DataFrame, recalls: list[str] | None = None, answer: bool = True) -> list:
    """The question after a one-line preamble. With recalls, the times of the entries the memory
    chose follow the question as text. Without answer, the prompt ends with the question."""
    recalls = recalls or [""] * len(rows)
    out = []
    for q, r in zip(rows.itertuples(), recalls, strict=True):
        n = q.context_length_samples
        post = q.question + (f" Recalled from memory: {r}." if r else "")
        out.append(
            {
                "pre_prompt": f"You are given a two-lead ECG recording of {duration(n)}, held in "
                "memory. Times are given from the start of the recording.",
                "time_series_text": [],
                "time_series": [],
                "post_prompt": post + " Answer:" if answer else q.question,
            }
        )
    return out


def tokens(bank: Bank, memory) -> tuple[torch.Tensor, torch.Tensor, list[list[str]]]:
    """Span and name of each token the blocks read after the null: the bank's slots (entries,
    then items), or with mean pooling each chunk from the earliest start to the latest end of its
    slots, unnamed."""
    start = torch.cat([bank.start, bank.item_start], 1)
    end = torch.cat([bank.end, bank.item_end], 1)
    inside = None if memory.chunk_of is None else memory.chunk_of[:, :, 1:]
    if inside is None or not inside.shape[-1]:
        # Without pooling, or an empty bank, whose chunks hold nothing to recall.
        return start, end, bank.names
    lo = torch.where(inside, start[:, None], torch.inf).amin(-1)
    hi = torch.where(inside, end[:, None], -torch.inf).amax(-1)
    n = inside.shape[1]
    return lo.nan_to_num(0, 0, 0), hi.nan_to_num(0, 0, 0), [[""] * n for _ in range(len(lo))]


def recall_text(start: list[float], end: list[float], names: list[str], chosen: list[int]) -> str:
    """Chosen tokens in time order as the benchmark's clock. Events carry their label, and spans
    that touch or overlap merge into one."""
    spans = []
    for j in sorted(chosen, key=lambda j: (start[j], end[j])):
        a, e, label = round(start[j] * HZ), round(end[j] * HZ), names[j]
        if not label and spans and not spans[-1][2] and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([a, e, label])
    out = []
    for a, e, label in spans:
        t = clock(a) if a == e else f"{clock(a)}-{clock(e)}"
        out.append(f"{label} {t}" if label else t)
    return "; ".join(out)


@torch.no_grad()
def recall(host, memory, rows: pd.DataFrame, bank: Bank, k: int, teacher: np.ndarray) -> list[str]:
    """The k tokens the memory attends to most at the question's last token, as text. Rows
    marked teacher take their evidence first, so the answer can learn to read the recall before
    the retrieval is good."""
    memory.load(bank)
    batch_ = prompts(rows, answer=False)
    embeds, mask = host.pad_and_apply_batch(batch_)
    lengths = mask.sum(1)
    post = [len(host.tokenizer(b["post_prompt"]).input_ids) for b in batch_]
    memory.start = lengths - torch.tensor(post, device=lengths.device)
    host.llm.model(inputs_embeds=embeds, attention_mask=mask)
    b = torch.arange(len(rows))
    attn = torch.stack([blk.last_attn[b, :, lengths - 1].mean(1) for blk in memory.blocks]).mean(0)
    evidence = memory.evidence(bank.slots).float()
    score = attn + torch.from_numpy(teacher).to(attn.device)[:, None] * evidence
    real = memory.mask[:, 1:]
    score = score[:, 1:].masked_fill(~real, -1.0)
    start, end, names = tokens(bank, memory)
    out = []
    for i in range(len(rows)):
        n = int(real[i].sum())
        chosen = score[i].topk(min(k, n)).indices.tolist() if n else []
        out.append(recall_text(start[i].tolist(), end[i].tolist(), names[i], chosen))
    return out


@dataclass
class Out:
    loss: torch.Tensor
    retrieval: torch.Tensor
    exact: torch.Tensor
    mass: torch.Tensor  # [B] memory attention on the evidence, mean over blocks


def forward(host, memory, rows: pd.DataFrame, bank: Bank, recalls) -> Out:
    """Answer loss, and the retrieval loss where the memory chooses: at the first answer position,
    or with recall at the question's last token, which the recall text cannot change."""
    memory.load(bank)
    batch_ = prompts(rows, recalls)
    logits, targets, sample, first = answer_logits(host, batch_, list(rows.answer), memory)
    loss, exact = loss_and_exact(logits, targets, sample, len(rows))
    if recalls is not None:
        q = [len(host.tokenizer(x).input_ids) for x in rows.question]
        first = memory.start + torch.tensor(q, device=first.device) - 1
    b = torch.arange(len(rows))
    attn = torch.stack([blk.last_attn[b, :, first].mean(1) for blk in memory.blocks])
    at = (attn * memory.evidence(bank.slots)).sum(-1)
    return Out(loss, -torch.log(at + 1e-6).mean(), exact, at.mean(0).detach())


@torch.no_grad()
def generate(host, memory, rows: pd.DataFrame, bank: Bank, recalls, max_new=24):
    """Greedy answers, with the whole sequence run again for each token. The blocks act from the
    question onward, which a key-value cache would not keep."""
    memory.load(bank)
    tok, embed = host.tokenizer, host.llm.get_input_embeddings()
    batch_ = prompts(rows, recalls)
    embeds, mask = host.pad_and_apply_batch(batch_)
    lengths = mask.sum(1)
    post = [len(tok(b["post_prompt"]).input_ids) for b in batch_]
    memory.start = lengths - torch.tensor(post, device=lengths.device)
    bos = embed(torch.tensor([tok.bos_token_id], device=embeds.device))
    seqs = [torch.cat([embeds[i, : lengths[i]], bos]) for i in range(len(rows))]
    out = [[] for _ in seqs]
    done = [False] * len(seqs)
    for _ in range(max_new):
        inputs = pad_sequence(seqs, batch_first=True)
        att = pad_sequence([torch.ones(len(s), dtype=torch.long) for s in seqs], batch_first=True)
        h = host.llm.model(
            inputs_embeds=inputs, attention_mask=att.to(inputs.device)
        ).last_hidden_state
        last = h[torch.arange(len(seqs)), torch.tensor([len(s) - 1 for s in seqs])]
        nxt = host.llm.lm_head(last).argmax(-1)
        for i, t in enumerate(nxt.tolist()):
            if done[i]:
                continue
            if t == tok.eos_token_id:
                done[i] = True
            else:
                out[i].append(t)
            seqs[i] = torch.cat([seqs[i], embed(nxt[i : i + 1])])
        if all(done):
            break
    return [tok.decode(o).strip() for o in out]


@torch.no_grad()
def score(host, memory, data: Data, idx: np.ndarray, kind: str, texts, vectors, k: int):
    """Per question: right by the benchmark's rules, the answer, memory mass on the evidence,
    and the recall text when k is set."""
    right, preds, mass, recalled = [], [], [], []
    step = 32 if kind == "none" else 8
    for i in range(0, len(idx), step):
        part = idx[i : i + step]
        rows = data.q.iloc[part]
        bank = data.bank(part, kind, texts, vectors, host.device)
        recalls = None
        if k:
            recalls = recall(host, memory, rows, bank, k, np.zeros(len(part)))
            recalled += recalls
        p = generate(host, memory, rows, bank, recalls)
        o = forward(host, memory, rows, bank, recalls)
        preds += p
        mass.append(o.mass.cpu().numpy())
        right += [
            correct(g, a, t) for g, a, t in zip(rows.answer, p, rows.answer_type, strict=True)
        ]
    return np.array(right), preds, np.concatenate(mass), recalled


def table(data: Data, idx: np.ndarray, right: np.ndarray, mass: np.ndarray) -> str:
    rows = data.q.iloc[idx]
    lines = []
    for t in TASKS:
        cells = []
        for s in sorted(rows.length.unique()):
            sel = ((rows.task_type == t) & (rows.length == s)).to_numpy()
            if sel.any():
                cells.append(f"{s // 60:>3d}m {right[sel].mean():.2f} ({mass[sel].mean():.2f})")
        if cells:
            lines.append(f"    {t:20s} " + " | ".join(cells))
    return f"  all {right.mean():.3f}\n" + "\n".join(lines)


@torch.no_grad()
def text_vectors(host, data: list[Data]) -> tuple[dict, torch.Tensor]:
    """Every event item text of the splits as the frozen LLM's mean last hidden state."""
    keys = set()
    for d in data:
        for bouts, beats in d.events.values():
            keys |= {("bout", lab, o) for lab, o in zip(bouts.label, bouts.ordinal, strict=True)}
            keys |= {("beat", lab, o) for lab, o in zip(beats.label, beats.ordinal, strict=True)}
    keys = sorted(keys)
    texts = [item_text(*k) for k in keys]
    out = []
    for i in range(0, len(texts), 512):
        t = host.tokenizer(texts[i : i + 512], return_tensors="pt", padding=True).to(host.device)
        h = host.llm.model(**t).last_hidden_state.float()
        m = t.attention_mask[..., None].float()
        out.append(((h * m).sum(1) / m.sum(1)).half().cpu())
    return {k: i for i, k in enumerate(keys)}, torch.cat(out)


def cells_sample(q: pd.DataFrame, per_cell: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for _, g in q.groupby(["task_type", "length"]):
        idx.append(rng.choice(g.index.to_numpy(), min(per_cell, len(g)), replace=False))
    return np.sort(np.concatenate(idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["events", "tiles", "none"], required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[900, 3600, 7200])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gate-lr", type=float, default=1e-2)
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--lora-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--retrieval-weight", type=float, default=10.0)
    ap.add_argument(
        "--recall", type=int, default=0, help="entries recalled into the prompt as text"
    )
    ap.add_argument(
        "--compress",
        type=int,
        default=0,
        help="time chunks whose mean slots the blocks read instead of the entries",
    )
    ap.add_argument(
        "--perceive",
        type=float,
        default=0.0,
        help="weight of a loss that predicts each train 10 s entry's rhythm from what it holds",
    )
    ap.add_argument(
        "--teacher",
        type=float,
        default=0.5,
        help="share of training recalls with the evidence first",
    )
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-per-cell", type=int, default=24)
    ap.add_argument("--test", action="store_true", help="score the test split at the end")
    ap.add_argument(
        "--score", action="store_true", help="score the test split with this run's saved weights"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True, help="run directory under runs/haystack")
    args = ap.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.txt", "a" if args.score else "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(str(vars(args)))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    base = Base() if args.bank == "tiles" else None
    splits = {s: Data(s, args.lengths, base) for s in ["train", "validation", "test"]}
    if args.perceive and args.bank != "tiles":
        raise ValueError("the perception loss needs tile entries")
    train, val = splits["train"], splits["validation"]
    train.perceive = bool(args.perceive)
    log(", ".join(f"{s} {len(d.q)} questions, {len(d.records)} records" for s, d in splits.items()))
    host = load_host()
    hidden = host.llm.config.hidden_size
    for d in splits.values():
        d.hidden = hidden
    texts, vectors = {}, None
    if args.bank == "events":
        texts, vectors = text_vectors(host, list(splits.values()))
        log(f"{len(texts)} event item texts")
    lora = add_lora(host.llm, args.lora_rank) if args.lora_rank else []
    layers = host.llm.model.layers
    memory = Memory(
        HAYSTACK, train.dim, N_STATS, hidden, len(layers) // 4, compress=args.compress
    ).to(host.device)
    at = memory.attach(layers)
    gates = [p for n, p in memory.named_parameters() if n.endswith("_gate")]
    rest = [p for n, p in memory.named_parameters() if not n.endswith("_gate")]
    # Nothing else supervises what an entry holds, only where the memory looks. The head reads
    # what the blocks would read before the span is added, so it acts under compression too.
    head = torch.nn.Linear(hidden, len(RHYTHMS)).to(host.device)
    extra = list(head.parameters()) if args.perceive else []
    groups = [{"params": rest + extra}, {"params": gates, "lr": args.gate_lr}]
    if lora:
        groups.append({"params": lora, "lr": args.lora_lr})
    trainable = [p for g in groups for p in g["params"]]
    n_mem = sum(p.numel() for p in memory.parameters())
    n_lora = sum(p.numel() for p in lora)
    log(f"memory blocks before layers {at}, {n_mem / 1e6:.1f}M memory and {n_lora / 1e6:.2f}M LoRA")
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * s / args.steps)),
    )
    cells = [g.index.to_numpy() for _, g in train.q.groupby(["task_type", "length"])]
    val_idx = cells_sample(val.q, args.eval_per_cell, np.random.default_rng(1234))
    banks = [args.bank] if args.bank == "none" else [args.bank, "none"]
    if args.bank == "tiles":
        banks.append("swap")

    def run_eval(s: int) -> None:
        host.llm.eval()
        memory.eval()
        log(f"eval step {s} validation, right per task and length (memory mass on the evidence)")
        for b in banks:
            right, preds, mass, recalls = score(
                host, memory, val, val_idx, b, texts, vectors, args.recall
            )
            log(f"  {b}\n" + table(val, val_idx, right, mass))
            # A few recalls of the trained bank beside the true answer, to see if they are tight.
            for j in range(0, len(recalls) if b == args.bank else 0, max(1, len(recalls) // 8)):
                q = val.q.iloc[val_idx[j]]
                log(
                    f"    recall {q.task_type} {q.length // 60}m: {recalls[j]} | "
                    f"answer {q.answer} | said {preds[j]}"
                )
        memory.train()
        host.llm.train()

    if args.score:
        w = torch.load(out / "weights.pt", map_location=host.device)
        memory.load_state_dict(w["memory"])
        assert not host.llm.load_state_dict(w["lora"], strict=False).unexpected_keys
    else:
        run_eval(0)
        host.llm.train()
        t0, run = time.time(), {"loss": [], "retrieval": [], "exact": [], "perceive": []}
        for s in range(1, args.steps + 1):
            idx = np.array(
                [rng.choice(cells[k]) for k in rng.integers(len(cells), size=args.batch)]
            )
            opt.zero_grad()
            for c in range(0, args.batch, args.micro):
                part = idx[c : c + args.micro]
                share = len(part) / args.batch
                rows = train.q.iloc[part]
                bank = train.bank(part, args.bank, texts, vectors, host.device)
                recalls = None
                if args.recall:
                    teacher = rng.random(len(part)) < args.teacher
                    recalls = recall(host, memory, rows, bank, args.recall, teacher)
                o = forward(host, memory, rows, bank, recalls)
                loss = o.loss + args.retrieval_weight * o.retrieval
                if args.perceive:
                    sel = bank.label >= 0
                    if sel.any():
                        logits = head(memory.content[sel])
                        ce = torch.nn.functional.cross_entropy(logits, bank.label[sel])
                        loss = loss + args.perceive * ce
                        acc = (logits.argmax(-1) == bank.label[sel]).float().mean()
                        run["perceive"].append(acc.item())
                (loss * share).backward()
                run["loss"].append(o.loss.item())
                run["retrieval"].append(o.retrieval.item())
                run["exact"].append(o.exact.float().mean().item())
            grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            if s % 50 == 0 or s == 1:
                g = " ".join(f"{torch.tanh(b.attn_gate).item():+.3f}" for b in memory.blocks)
                gb = torch.cuda.max_memory_allocated() / 2**30
                log(
                    f"step {s} loss {np.mean(run['loss']):.4f} retrieval "
                    f"{np.mean(run['retrieval']):.3f} exact {np.mean(run['exact']):.3f} "
                    f"grad {grad:.2f} gates {g} {(time.time() - t0) / s:.2f}s/step {gb:.1f}GB"
                    + (
                        f" rhythm accuracy {np.mean(run['perceive']):.3f}"
                        if run["perceive"]
                        else ""
                    )
                )
                run = {k: [] for k in run}
            if s % args.eval_every == 0 or s == args.steps:
                run_eval(s)
        lora_state = {n: p for n, p in host.llm.named_parameters() if "lora_" in n}
        torch.save({"memory": memory.state_dict(), "lora": lora_state}, out / "weights.pt")
    if not (args.test or args.score):
        return
    host.llm.eval()
    memory.eval()
    test = splits["test"]
    idx = np.arange(len(test.q))
    res = {
        "task": test.q.task_type.to_numpy().astype(str),
        "length": test.q.length.to_numpy(),
        "record": test.q.record_id.to_numpy().astype(str),
    }
    log("test, right per task and length (memory mass on the evidence)")
    for b in banks:
        right, preds, mass, _ = score(host, memory, test, idx, b, texts, vectors, args.recall)
        res[f"{b}_right"], res[f"{b}_mass"] = right, mass
        res[f"{b}_pred"] = np.array(preds, dtype=str)
        log(f"  {b}\n" + table(test, idx, right, mass))
    np.savez(out / "test.npz", **res)


if __name__ == "__main__":
    main()
