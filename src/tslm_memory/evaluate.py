"""Score trained London runs on the test households: exact match per family and bank, plus
within one for the numeric families, from the first answer token (numbers below 1000 are one
Llama token). Per-question results go to the run's test.npz for the report."""

import argparse
import ast

import numpy as np
import torch

from tslm_memory.train import BANKS, RUNS, answer, build, chance, draw, load_weights, open_split

NUMERIC = ["recall", "last_time", "count"]


def gaps(first: np.ndarray, qs: list, tok) -> np.ndarray:
    """Distance of the predicted first token from the answer for the numeric families, else 99."""
    out = []
    for q, p in zip(qs, first, strict=True):
        text = tok.decode([int(p)]).strip()
        out.append(abs(int(text) - int(q.answer)) if q.family in NUMERIC and text.isdigit() else 99)
    return np.array(out)


def scores(exact: np.ndarray, gap: np.ndarray, qs: list) -> str:
    fam = np.array([q.family for q in qs])
    parts = []
    for f in dict.fromkeys(fam):
        s = f"{f} {exact[fam == f].mean():.3f}"
        if f in NUMERIC:
            s += f" ({(gap[fam == f] <= 1).mean():.2f})"
        parts.append(s)
    return " | ".join(parts)


def ages(qs: list) -> np.ndarray:
    """Days back to the first evidence day, 0 for questions without day evidence."""
    return np.array([q.now - q.evidence[0] if q.evidence else 0 for q in qs])


def by_age(exact: np.ndarray, mass: np.ndarray, qs: list) -> str:
    """Exact match (and memory attention on the evidence) of recall and last time by how far
    back the evidence day lies."""
    fam = np.array([q.family for q in qs])
    age = ages(qs)
    parts = []
    for f in ["recall", "last_time"]:
        for lo, hi in [(1, 30), (31, 100), (101, 200), (201, 400), (401, 600), (601, 10_000)]:
            sel = (fam == f) & (age >= lo) & (age <= hi)
            if sel.any():
                parts.append(
                    f"{f} {lo}-{hi} {exact[sel].mean():.2f} ({mass[sel].mean():.2f}, n {sel.sum()})"
                )
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories under runs/london")
    ap.add_argument("--questions", type=int, default=512, help="per family")
    args = ap.parse_args()
    print(f"test: exact per family, (within one) for {', '.join(NUMERIC)}")
    sp = open_split("test")
    for run in args.runs:
        out = RUNS / run
        # The first line of a run's log holds its arguments.
        cfg = ast.literal_eval(open(out / "log.txt").readline())
        window, families = cfg["window"], cfg["families"]
        qs = draw(sp, args.questions * len(families), np.random.default_rng(4321), window, families)
        print(chance(qs))
        host, _, memory, _ = build(cfg["lora_rank"], [sp], cfg["compress"])
        memory.eval()
        load_weights(host, memory, out / "weights.pt")
        print(f"{run} (LoRA rank {cfg['lora_rank']}, trained {cfg['mode']}, window {window})")
        # Per question, for intervals over households in the report.
        rows = {
            "family": np.array([q.family for q in qs]),
            "household": np.array([q.household for q in qs]),
            "answer": np.array([q.answer for q in qs]),
            "age": ages(qs),
        }
        # A run trained without memory is scored without memory.
        for bank in ["none"] if cfg["mode"] == "none" else BANKS:
            a = answer(host, memory, sp, qs, bank, window)
            gap = gaps(a.first, qs, host.tokenizer)
            print(f"  {bank:6s} " + scores(a.exact, gap, qs), flush=True)
            if bank == "full":
                print("    by days before now: " + by_age(a.exact, a.mass, qs), flush=True)
            rows |= {f"{bank}_exact": a.exact, f"{bank}_gap": gap, f"{bank}_mass": a.mass}
        np.savez(out / "test.npz", **rows)
        del host, memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
