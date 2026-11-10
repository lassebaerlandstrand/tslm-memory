"""Train the memory on Low Carbon London household days.

--mode full trains with each household's history in memory, --mode none with an empty memory,
which is the no-memory baseline. --compress is the compression baseline."""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tslm_memory.encode import N_STATS, RULES, features, raw_stats
from tslm_memory.lcl import DATA, load
from tslm_memory.memory import Bank, Memory, Part, Settings, batch, pool_stats, pool_vectors
from tslm_memory.questions import FAMILIES, WEEKDAYS, Households, Question, ask, items, weekday
from tslm_memory.tslm import add_lora, answer_logits, load_host, loss_and_exact, prompts

RUNS = Path("runs/london")
BANKS = ["oracle", "full", "swap", "none"]
# London's settings for the shared memory: a clock of days before now with a calendar, the
# calendar pyramid of day, week and month, and every entry's whole total as an exact value.
LONDON = Settings(levels=3, periods=(2, 2048, 16), calendar=True, exact=1)


@dataclass
class Split:
    h: Households
    entries: np.ndarray  # [rows, 3, d] fp16 Chronos-2 patches, memory-mapped, rows as in h.offset
    stats: np.ndarray  # [rows, N_STATS] day features
    total: np.ndarray  # [rows] day totals in kWh, the exact value
    # Week and month entries, rows as in h.period_offset.
    period_entries: np.ndarray  # [rows, 3, d]
    period_stats: np.ndarray  # [rows, N_STATS]
    period_total: np.ndarray  # [rows]
    notices: torch.Tensor | None = None  # [schedule days, hidden] price notices, set by build


def open_split(name: str) -> Split:
    h = load(name)
    entries = np.load(DATA / f"{name}_chronos.npy", mmap_mode="r")
    raw = [raw_stats(np.asarray(d)) for d in h.days]
    path = DATA / f"{name}_periods.npy"
    if not path.exists():
        rows = [
            pool_vectors(torch.from_numpy(np.asarray(entries[a:b])), p).half().numpy()
            for a, b, p in zip(h.offset[:-1], h.offset[1:], h.periods, strict=True)
        ]
        np.save(path, np.concatenate(rows))
    pooled = np.concatenate([pool_stats(r, RULES, p) for r, p in zip(raw, h.periods, strict=True)])
    per = np.concatenate(h.periods)
    return Split(
        h,
        entries,
        features(np.concatenate(raw), 1),
        np.concatenate(raw)[:, 0],
        np.load(path),
        features(pooled, per[:, 2] - per[:, 1] + 1),
        pooled[:, 0],
    )


def swapped(sp: Split, q: Question) -> tuple[int, int]:
    """The next household and the day that stands for now in it: the same day index, or its last
    day if it is shorter, so its span before now has the same days before now."""
    owner = (q.household + 1) % len(sp.h.days)
    return owner, min(q.now, len(sp.h.days[owner]) - 1)


def memory_for(sp: Split, qs: list[Question], bank: str, window: int, device: str) -> Bank:
    """The bank for a batch, and per sample the memory slots that hold the evidence.

    full: the days in [now - window, now), the weeks and months that end in that span and the
    span's price notices. oracle: only the evidence. swap: the same span of the next household,
    so nothing in it answers the question. none: only the null entry. Entry slots hold the days
    and then the weeks and months, each spanning its days on a clock of days before now. The null
    entry counts as the evidence for questions without evidence."""
    entries, notes, null = [], [], []
    width = int(np.prod(sp.entries.shape[1:]))
    for q in qs:
        owner, now, lo = q.household, q.now, max(0, q.now - window)
        if bank == "swap":
            owner, now = swapped(sp, q)
            lo = max(0, lo - (q.now - now))
        per = sp.h.periods[owner]
        if bank == "oracle":
            days, pi, it = (np.array(x, int) for x in (q.evidence, q.periods, q.items))
        elif bank == "none":
            days = pi = it = np.arange(0)
        else:
            days = np.arange(lo, now)
            pi = np.flatnonzero((per[:, 1] >= lo) & (per[:, 2] < now))
            it = items(sp.h, owner, lo, now)
        own = bank in ("oracle", "full")
        rows, prow = sp.h.offset[owner] + days, sp.h.period_offset[owner] + pi
        first = np.concatenate([days, per[pi, 1]])
        last = np.concatenate([days, per[pi, 2]])
        vectors = np.concatenate([sp.entries[rows], sp.period_entries[prow]])
        total = np.concatenate([sp.total[rows], sp.period_total[prow]])
        entries.append(
            Part(
                torch.from_numpy(vectors.reshape(len(vectors), width)),
                (now - first).astype(np.float32),
                (now - last - 1).astype(np.float32),
                np.concatenate([np.isin(days, q.evidence), np.isin(pi, q.periods)]) & own,
                stats=np.concatenate([sp.stats[rows], sp.period_stats[prow]]),
                exact=total[:, None],
                level=np.concatenate([np.zeros(len(days), int), per[pi, 0]]),
                date=sp.h.first[owner] + first,
            )
        )
        d = sp.h.first[owner] + it
        vec = sp.notices[d - sp.h.prices.first] if len(it) else torch.zeros(0, 1)
        notes.append(
            Part(
                vec,
                (now - it).astype(np.float32),
                (now - it - 1).astype(np.float32),
                np.isin(it, q.items) & own,
                date=d,
            )
        )
        null.append(not q.evidence and not q.items and not q.periods)
    hidden = sp.notices.shape[1] if sp.notices is not None else 1
    return batch(entries, notes, null, (width, N_STATS, LONDON.exact, hidden), device)


@dataclass
class Out:
    loss: torch.Tensor
    retrieval: torch.Tensor
    exact: torch.Tensor  # [B]
    mass: torch.Tensor  # [B] memory attention on the evidence at the answer, mean over blocks
    first: torch.Tensor  # [B] predicted first answer token


def forward(host, memory: Memory, sp: Split, qs: list[Question], bank: Bank) -> Out:
    memory.load(bank)
    days = []
    for q in qs:
        name = WEEKDAYS[weekday(sp.h.first[q.household] + q.now)]
        today = torch.from_numpy(np.array(sp.h.days[q.household][q.now])).cuda()
        days.append((f"This is today's half-hourly use in kWh, on a {name}", today))
    logits, targets, sample, first = answer_logits(
        host, prompts([q.text for q in qs], days), [q.answer for q in qs], memory
    )
    loss, exact = loss_and_exact(logits, targets, sample, len(qs))
    starts = torch.ones_like(sample, dtype=torch.bool)
    starts[1:] = sample[1:] != sample[:-1]
    pred = logits[starts].argmax(-1)
    # Memory attention at the position that predicts the first answer token, mean over heads
    # [blocks, B, slots], supervised on the evidence there. Supervised at question tokens, LoRA
    # did not carry it on.
    b = torch.arange(len(qs))
    attn = torch.stack([blk.last_attn[b, :, first].mean(1) for blk in memory.blocks])
    at_answer = (attn * memory.evidence(bank.slots)).sum(-1)
    retrieval = -torch.log(at_answer + 1e-6).mean()
    return Out(loss, retrieval, exact, at_answer.mean(0).detach(), pred)


def draw(sp: Split, n: int, rng: np.random.Generator, window: int, families: list[str]):
    return [ask(sp.h, families[i % len(families)], rng, window) for i in range(n)]


@dataclass
class Answers:
    exact: np.ndarray  # [Q]
    first: np.ndarray  # [Q] predicted first answer token
    mass: np.ndarray  # [Q] memory attention on the evidence at the answer


@torch.no_grad()
def answer(host, memory: Memory, sp: Split, qs: list[Question], bank: str, window: int) -> Answers:
    parts = {k: [] for k in ["exact", "first", "mass"]}
    for i in range(0, len(qs), 32):
        part = qs[i : i + 32]
        o = forward(host, memory, sp, part, memory_for(sp, part, bank, window, "cuda"))
        parts["exact"].append(o.exact)
        parts["first"].append(o.first)
        parts["mass"].append(o.mass)
    return Answers(**{k: torch.cat(v).cpu().numpy() for k, v in parts.items()})


def evaluate(host, memory: Memory, sp: Split, qs: list[Question], bank: str, window: int) -> str:
    """Exact per family with memory attention on the evidence."""
    a = answer(host, memory, sp, qs, bank, window)
    fam = np.array([q.family for q in qs])
    parts = [
        f"{f} {a.exact[fam == f].mean():.3f} ({a.mass[fam == f].mean():.2f})"
        for f in dict.fromkeys(fam)
    ]
    return f"{bank:6s} " + " | ".join(parts)


def chance(qs: list[Question]) -> str:
    """Share of the most common answer per family."""
    parts = []
    for f in dict.fromkeys(q.family for q in qs):
        answers = [q.answer for q in qs if q.family == f]
        parts.append(f"{f} {max(map(answers.count, set(answers))) / len(answers):.3f}")
    return "chance " + " | ".join(parts)


@torch.no_grad()
def encode_notices(host, sp: Split) -> torch.Tensor:
    """Every schedule day's price notice as the frozen LLM's mean last hidden state over its
    tokens, zeros for normal days. Run before LoRA and the memory blocks are attached."""
    p = sp.h.prices
    texts = [p.notice(p.first + r) for r in range(len(p.grid))]
    out = torch.zeros(len(texts), host.llm.config.hidden_size, dtype=torch.float16)
    keep = [r for r, t in enumerate(texts) if t]
    tok = host.tokenizer([texts[r] for r in keep], return_tensors="pt", padding=True).to("cuda")
    h = host.llm.model(**tok).last_hidden_state.float()
    m = tok.attention_mask[..., None].float()
    out[keep] = ((h * m).sum(1) / m.sum(1)).half().cpu()
    return out


def build(lora_rank: int, splits: list[Split], compress: int):
    """Host, LoRA and memory blocks (reading compress time chunks if set), with the price
    notices of every split encoded."""
    host = load_host()
    for s in splits:
        s.notices = encode_notices(host, s)
    lora = add_lora(host.llm, lora_rank) if lora_rank else []
    layers = host.llm.model.layers
    memory = Memory(
        LONDON,
        splits[0].entries[0].size,
        N_STATS,
        host.llm.config.hidden_size,
        len(layers) // 4,
        compress=compress,
    ).cuda()
    return host, lora, memory, memory.attach(layers)


def load_weights(host, memory: Memory, path: Path) -> dict:
    """Weights of an earlier run. Returns the saved file, which holds the optimizer state."""
    weights = torch.load(path)
    memory.load_state_dict(weights["memory"])
    if weights["lora"]:
        assert not host.llm.load_state_dict(weights["lora"], strict=False).unexpected_keys
    return weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "none"], required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=200, help="days before now held in memory")
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--init", help="run under runs/london to continue from")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gate-lr", type=float, default=1e-2)
    ap.add_argument("--lora-rank", type=int, default=4, help="0 keeps the LLM frozen")
    ap.add_argument("--lora-lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-questions", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrieval-weight", type=float, default=10.0)
    ap.add_argument(
        "--compress",
        type=int,
        default=0,
        help="time chunks whose mean slots the blocks read instead of the entries",
    )
    ap.add_argument("--name", required=True, help="run directory under runs/london")
    args = ap.parse_args()
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out / "log.txt", "w")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val = open_split("train"), open_split("val")
    host, lora, memory, at = build(args.lora_rank, [train, val], args.compress)
    n_mem = sum(p.numel() for p in memory.parameters())
    n_lora = sum(p.numel() for p in lora)
    log(
        f"{vars(args)}\nmemory blocks before layers {at}, "
        f"{n_mem / 1e6:.1f}M memory and {n_lora / 1e6:.2f}M LoRA parameters"
    )

    # The gates scale every update to the blocks' output, so at the base rate they stay near 0.
    gates = [p for n, p in memory.named_parameters() if n.endswith("_gate")]
    rest = [p for n, p in memory.named_parameters() if not n.endswith("_gate")]
    groups = [{"params": rest}, {"params": gates, "lr": args.gate_lr}]
    if lora:
        groups.append({"params": lora, "lr": args.lora_lr})
    trainable = [p for g in groups for p in g["params"]]
    # beta2 0.95 bounds the step a sudden gradient spike can take after a calm stretch. With
    # 0.999 a run diverged at step 4700.
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if args.init:
        # A continuation keeps Adam's moments, so it does not restart from a fresh estimate.
        # Loading also brings the old rates, so this run's rates are put back before the schedule
        # takes them.
        saved = load_weights(host, memory, RUNS / args.init / "weights.pt")
        rates = [g["lr"] for g in opt.param_groups]
        opt.load_state_dict(saved["opt"])
        for g, lr in zip(opt.param_groups, rates, strict=True):
            g["lr"] = lr
            g.pop("initial_lr", None)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * s / args.steps)),
    )
    # One batch is drawn and dropped. The reported runs did the same, and without it a seed
    # would give other questions than theirs.
    draw(train, args.batch, rng, args.window, args.families)
    val_qs = draw(val, args.eval_questions, np.random.default_rng(1234), args.window, args.families)
    log(chance(val_qs))

    def run_eval(s: int) -> None:
        host.llm.eval()
        memory.eval()
        log(f"eval step {s} (exact, memory attention on the evidence at the answer)")
        for b in BANKS:
            log("  " + evaluate(host, memory, val, val_qs, b, args.window))
        memory.train()

    lora_b = [p for n, p in host.llm.named_parameters() if "lora_B" in n]
    run_eval(0)
    t0 = time.time()
    run = {k: [] for k in ["loss", "retrieval", "exact"]}
    zero = torch.zeros((), device="cuda")
    for s in range(1, args.steps + 1):
        qs = draw(train, args.batch, rng, args.window, args.families)
        opt.zero_grad()
        o = forward(host, memory, train, qs, memory_for(train, qs, args.mode, args.window, "cuda"))
        (o.loss + args.retrieval_weight * o.retrieval).backward()
        run["loss"].append(o.loss.item())
        run["retrieval"].append(o.retrieval.item())
        run["exact"].append(o.exact.float().mean().item())
        # With only the null entry in memory (--mode none) the entry projections get no gradient.
        norms = [
            torch.stack([p.grad.norm() for p in g["params"] if p.grad is not None] or [zero]).norm()
            for g in groups
        ]
        grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if s % 50 == 0 or s == 1:
            gates = " ".join(f"{torch.tanh(b.attn_gate).item():+.3f}" for b in memory.blocks)
            lora_norm = torch.stack([p.norm() for p in lora_b]).norm() if lora else 0.0
            gb = torch.cuda.max_memory_allocated() / 2**30
            group_norms = " ".join(f"{n:.1f}" for n in norms)
            log(
                f"step {s} loss {np.mean(run['loss']):.4f} retrieval "
                f"{np.mean(run['retrieval']):.3f} exact {np.mean(run['exact']):.3f} "
                f"grad {grad:.2f} ({group_norms}) gates {gates} lora B {lora_norm:.2f} "
                f"{(time.time() - t0) / s:.2f}s/step {gb:.1f}GB"
            )
            run = {k: [] for k in run}
        if s % args.eval_every == 0 or s == args.steps:
            run_eval(s)
    lora_state = {n: p for n, p in host.llm.named_parameters() if "lora_" in n}
    weights = {"memory": memory.state_dict(), "lora": lora_state, "opt": opt.state_dict()}
    torch.save(weights, out / "weights.pt")


if __name__ == "__main__":
    main()
