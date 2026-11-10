"""Collect the test results into LaTeX tables and a PDF figure under runs/report.

Reads the per-question results that evaluate and haystack write to test.npz in each run's
directory under runs/london and runs/haystack. Intervals are 95% from resampling households or
records, so questions of one household or record move together. Seeds that have not finished
yet are left out, and the captions give how many each row averages."""

import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

from tslm_memory.ltaf import TASKS, questions  # noqa: E402

LONDON = Path("runs/london")
HAYSTACK = Path("runs/haystack")
OUT = Path("runs/report")
FAMILIES = {
    "recall": "Day total",
    "last_time": "Last above X",
    "count": "Count",
    "baseline": "Usual day",
    "price_last": "Last high price",
    "month_change": "Month change",
    "month_total": "Month total",
    "control": "Today only",
}
# One training wording per family, with example numbers, for the London table.
EXAMPLES = {
    "recall": "What was the total electricity use 12 days before today, in kWh?",
    "last_time": "How many days before today did daily use last exceed 15 kWh?",
    "count": "On how many of the last 14 days did daily use exceed 9 kWh?",
    "baseline": "Was today's total use higher, lower or about the same as on a usual Monday "
    "over the last 8 weeks?",
    "price_last": "How many days before today was the last day with a high electricity price?",
    "month_change": "Was the total use last month higher, lower or about the same as in the "
    "same month a year before?",
    "month_total": "What was the total electricity use last month, in kWh to the nearest 10?",
    "control": "At which hour of the day, from 0 to 23, was today's use highest?",
}
BINS = [(1, 100), (101, 200), (201, 300), (301, 400), (401, 600), (601, 830)]
# Categorical slots of the validated default palette, and its text inks.
BLUE, AQUA = "#2a78d6", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


def bootstrap(
    values: np.ndarray, groups: np.ndarray, weights: np.ndarray | None = None, n: int = 2000
) -> tuple[float, float]:
    """95% interval of the (weighted) mean, resampling groups."""
    w = np.ones(len(values)) if weights is None else weights
    ids, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, w * values, len(ids))
    counts = np.bincount(inv, w, len(ids))
    draw = np.random.default_rng(0).integers(len(ids), size=(n, len(ids)))
    means = sums[draw].sum(1) / counts[draw].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(run: str) -> dict | None:
    path = LONDON / run / "test.npz"
    return dict(np.load(path)) if path.exists() else None


def bold_best(values: list[float]) -> list[str]:
    """Values to two decimals, the largest in bold (all of them on a tie)."""
    top = max(round(v, 2) for v in values)
    return [f"\\textbf{{{v:.2f}}}" if round(v, 2) == top else f"{v:.2f}" for v in values]


# A paragraph column with ragged right edges, for the example questions.
RAGGED = ">{\\raggedright\\arraybackslash}p"


def write_table(path: Path, spec: str, head: list[str], body: list[str], caption: str) -> None:
    """A LaTeX table labelled tab:<file name>, from ready-made header and body lines."""
    lines = [
        "\\begin{table}[!htb]",
        "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{path.stem}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        *head,
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(path)
    print("\n".join("  " + x for x in body))


def interval(d: np.ndarray, groups: np.ndarray, w: np.ndarray | None = None) -> str:
    m = d.mean() if w is None else (d * w).sum()
    lo, hi = bootstrap(d, groups, w)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def seed_runs(rs: list[dict | None], bank: str) -> list[np.ndarray]:
    """Per finished seed, a bank's exact match per question. The seeds must hold the same
    questions."""
    done = [r for r in rs if r is not None and f"{bank}_exact" in r]
    for r in done[1:]:
        same = all(np.array_equal(r[k], done[0][k]) for k in ("family", "household", "answer"))
        assert same, "seeds were scored on different questions"
    return [r[f"{bank}_exact"].astype(float) for r in done]


def seeds_note(counts: dict[str, int], spread: float) -> str:
    """The caption sentence on seeds: how many each system averages and how far seeds differ."""
    if len(set(counts.values())) == 1:
        n = next(iter(counts.values()))
        return (
            f"Each value is the mean over {n} seeds, whose scores differ by a standard deviation "
            f"of at most {spread:.2f}."
        )
    parts = ", ".join(f"{n} for {k[0].lower() + k[1:]}" for k, n in counts.items())
    return (
        f"Each value is the mean over seeds ({parts}), whose scores differ by a standard "
        f"deviation of at most {spread:.2f}."
    )


def seed_spread(scores: list[list[float]]) -> float:
    """The largest standard deviation over seeds of one score, for systems with several seeds.
    scores holds, per system and score, the per-seed values."""
    return max((float(np.std(v)) for v in scores if len(v) > 1), default=0.0)


def london_results(memory: list[str], compression: list[str], none: list[str], noret: str) -> dict:
    """Per system, per seed, exact match per question. Memory with another household's bank comes
    from the memory runs. All runs hold the same questions."""
    mem, comp, no = ([load(r) for r in runs] for runs in (memory, compression, none))
    ref = next(r for r in [*mem, *comp, *no] if r is not None)
    for r in [*mem, *comp, *no, load(noret)]:
        if r is not None:
            assert np.array_equal(r["household"], ref["household"]), "runs differ in questions"
    runs = {
        "Memory": seed_runs(mem, "full"),
        "Compression": seed_runs(comp, "full"),
        "No memory": seed_runs(no, "none"),
        "Another household's bank": seed_runs(mem, "swap"),
        "Trained without the retrieval loss": seed_runs([load(noret)], "full"),
    }
    systems = {k: np.mean(v, 0) for k, v in runs.items() if v}
    keep = ("family", "household", "age")
    return {k: ref[k] for k in keep} | {"systems": systems, "runs": runs}


def london_table(res: dict) -> None:
    """Families by row with an example question, memory, compression and no memory by column,
    and their average over the families. The differences the text cites, with 95% intervals over
    households, go to london_diffs.txt."""
    fam, household, systems = res["family"], res["household"], res["systems"]
    shown = ["Memory", "Compression", "No memory"]
    body = []
    for f, name in FAMILIES.items():
        cells = bold_best([systems[k][fam == f].mean() for k in shown])
        body.append(f"{name} & {EXAMPLES[f]} & " + " & ".join(cells) + " \\\\")
    # Every family has the same number of questions, so the mean over questions is the average.
    cells = bold_best([systems[k].mean() for k in shown])
    body += ["\\midrule", "Average & & " + " & ".join(cells) + " \\\\"]
    runs = res["runs"]
    spread = [
        [float(r[sel].mean()) for r in runs[k]]
        for k in shown
        for sel in [*(fam == f for f in FAMILIES), np.ones(len(fam), bool)]
    ]
    caption = (
        "London, exact match on 512 test questions per family from 200 households not seen in "
        "training. Each family has one wording in training, and the numbers in a question are "
        "drawn per question. The best result per row is in bold. "
        + seeds_note({k: len(runs[k]) for k in shown}, seed_spread(spread))
    )
    head = ["Family & Example question & Memory & Compression & No memory \\\\"]
    write_table(OUT / "london.tex", f"@{{}}l{RAGGED}{{7.6cm}}ccc@{{}}", head, body, caption)
    lines = []
    others = ["Compression", "No memory", "Another household's bank"]
    others.append("Trained without the retrieval loss")
    for other in others:
        d = systems["Memory"] - systems[other]
        lines.append(f"Memory - {other}, average: {interval(d, household)}")
        for f, name in FAMILIES.items():
            lines.append(f"Memory - {other}, {name}: {interval(d[fam == f], household[fam == f])}")
    for other in others:
        cells = [f"{name} {systems[other][fam == f].mean():.3f}" for f, name in FAMILIES.items()]
        lines.append(f"{other}: average {systems[other].mean():.3f}, " + ", ".join(cells))
    (OUT / "london_diffs.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


LENGTHS = {900: "15 min", 3600: "1 h", 7200: "2 h"}
GROUPS = {
    "Presence": ["existence", "anomaly_detection", "state_query"],
    "Order and time": [
        "ordering", "localization", "comparison", "multi_hop", "anomaly_localization",
    ],
    "Counting, antecedent": ["counting", "antecedent"],
}  # fmt: skip
# One test question per group, for the TS-Haystack table.
GROUP_EXAMPLES = {
    "Presence": "Is there any atrial bigeminy episode in this ECG window?",
    "Order and time": "Between which timestamps does the 5th atrial fibrillation episode fall?",
    "Counting, antecedent": "How many times does atrial bigeminy occur in this recording?",
}


def haystack_right(runs: list[str], bank: str) -> dict | None:
    """Per test question: task, length, record and right, the mean over the runs (seeds) that
    have finished, which must hold the same questions."""
    done = []
    for run in runs:
        path = HAYSTACK / run / "test.npz"
        if path.exists():
            r = dict(np.load(path))
            if f"{bank}_right" in r:
                done.append(r)
    if not done:
        return None
    for r in done[1:]:
        assert all(np.array_equal(r[k], done[0][k]) for k in ("task", "length", "record"))
    per_seed = [r[f"{bank}_right"].astype(float) for r in done]
    right = np.mean(per_seed, 0)
    keep = {k: done[0][k] for k in ("task", "length", "record")}
    return keep | {"right": right, "n": len(done), "per_seed": per_seed}


def cell_weights(r: dict, tasks: list[str]) -> np.ndarray:
    """Per question, a weight that gives every task and length cell of the group the same
    share, so a group's score is the mean over its cells as in the paper."""
    w = np.zeros(len(r["task"]))
    cells = [(t, s) for t in tasks for s in np.unique(r["length"])]
    cells = [c for c in cells if ((r["task"] == c[0]) & (r["length"] == c[1])).any()]
    for t, s in cells:
        sel = (r["task"] == t) & (r["length"] == s)
        w[sel] = 1 / sel.sum() / len(cells)
    return w


def haystack_table(systems: dict[str, tuple[list[str], str]]) -> dict:
    """Task groups by row with an example question, memory and compression on each bank and no
    memory by column, then the average over all task and length cells as in the paper. Bold is
    the better of memory and compression on each bank. The differences the text cites go to
    haystack_diffs.txt. Returns the results found, for the figure."""
    res = {name: haystack_right(runs, bank) for name, (runs, bank) in systems.items()}
    res = {k: v for k, v in res.items() if v is not None}
    order = [
        "Memory, annotated events", "Compression, annotated events",
        "Memory, ECG segments", "Compression, ECG segments", "No memory",
    ]  # fmt: skip
    if not all(k in res for k in order):
        print("TS-Haystack: runs missing", [k for k in order if k not in res])
        return res
    ref = res["Memory, ECG segments"]

    spread = []

    def cells(tasks: list[str]) -> list[str]:
        w = cell_weights(ref, tasks)
        v = [(res[k]["right"] * w).sum() for k in order]
        spread.extend([float((r * w).sum()) for r in res[k]["per_seed"]] for k in order)
        return [*bold_best(v[:2]), *bold_best(v[2:4]), f"{v[4]:.2f}"]

    body = [
        f"{g} & {GROUP_EXAMPLES[g]} & " + " & ".join(cells(t)) + " \\\\" for g, t in GROUPS.items()
    ]
    body += ["\\midrule", "Average & & " + " & ".join(cells(TASKS)) + " \\\\"]
    head = [
        "& & \\multicolumn{2}{c}{Annotated events} & \\multicolumn{2}{c}{ECG segments} & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Tasks & Example question & Memory & Compression & Memory & Compression & No memory \\\\",
    ]
    caption = (
        "TS-Haystack, accuracy by the benchmark's scoring on recordings of 15 min, 1 h and 2 h "
        "from 9 test patients, as the mean over task and length cells. Presence is existence, "
        "anomaly detection and state query. Order and time is ordering, localization, "
        "comparison, multi-hop and anomaly localization, which ask about an event by its order "
        "or ask for its time. Antecedent asks which rhythm came right before an event. The "
        "memory holds either the recording's annotated events or the ECG itself in 10 s "
        "segments. Every system's prompt holds only the question. The better of memory and "
        "compression on each kind of memory is in bold. "
        + seeds_note({k: res[k]["n"] for k in order}, seed_spread(spread))
    )
    write_table(OUT / "haystack.tex", f"@{{}}l{RAGGED}{{4.1cm}}ccccc@{{}}", head, body, caption)
    lines = []
    for a, b in [
        ("Memory, ECG segments", "Compression, ECG segments"),
        ("Memory, annotated events", "Compression, annotated events"),
        ("Memory, ECG segments", "No memory"),
    ]:
        d = res[a]["right"] - res[b]["right"]
        for g, tasks in [*GROUPS.items(), ("All", TASKS)]:
            lines.append(f"{a} - {b}, {g}: {interval(d, ref['record'], cell_weights(ref, tasks))}")
    (OUT / "haystack_diffs.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def mass_on_evidence(runs: list[dict], sel: np.ndarray, bank: str = "full") -> float:
    """Memory attention on the evidence, over the selected questions of every seed."""
    return float(np.concatenate([r[f"{bank}_mass"][sel] for r in runs]).mean())


def discussion(memory: list[str], london: dict, haystack: dict[str, dict]) -> None:
    """The error analysis the discussion cites, to discussion.txt. On London: where the memory
    looks when it is wrong, how far off a wrong day total is, and why day total dips by days
    back. On TS-Haystack: how much of ordering the question's ordinals answer."""
    runs = [r for r in map(load, memory) if r is not None]
    fam, age = london["family"], london["age"]
    # The oracle bank holds only the evidence. Not every memory run evaluated it.
    oracle = [r for r in runs if "oracle_exact" in r]
    total = np.array(
        [float(a) if f == "recall" else np.nan for f, a in zip(fam, runs[0]["answer"], strict=True)]
    )

    def only_evidence(sel: np.ndarray) -> float:
        return float(np.mean([r["oracle_exact"][sel].mean() for r in oracle]))

    lines = []
    for f in ["recall", "last_time", "price_last"]:
        wrong = [(fam == f) & ~r["full_exact"] for r in runs]
        mass = np.concatenate([r["full_mass"][w] for r, w in zip(runs, wrong, strict=True)])
        lines.append(
            f"{FAMILIES[f]}: memory {london['systems']['Memory'][fam == f].mean():.3f}, "
            f"attention on the evidence when wrong {mass.mean():.3f}, "
            f"bank with only the evidence {only_evidence(fam == f):.3f}"
        )
    day = fam == "recall"
    gaps = np.concatenate([r["full_gap"][day & ~r["full_exact"]] for r in runs])
    lines.append(f"Day total, wrong answers within 1 kWh: {(np.abs(gaps) <= 1).mean():.3f}")
    memory_day = london["systems"]["Memory"]
    for lo, hi in [(0, 10), (10, 20), (20, np.inf)]:
        sel = day & (total >= lo) & (total < hi)
        lines.append(
            f"Day total {lo}-{hi} kWh: memory {memory_day[sel].mean():.3f} (n {sel.sum()})"
        )
    for lo, hi in BINS:
        sel = day & (age >= lo) & (age <= hi)
        lines.append(
            f"Day total {lo}-{hi} days back: memory {memory_day[sel].mean():.3f}, "
            f"below 20 kWh {memory_day[sel & (total < 20)].mean():.3f}, "
            f"attention on the evidence {mass_on_evidence(runs, sel):.3f}, "
            f"bank with only the evidence {only_evidence(sel):.3f}, "
            f"totals of 20 kWh or more {(total[sel] >= 20).mean():.3f}, "
            f"n {sel.sum()}, households {len(np.unique(london['household'][sel]))}"
        )
    # Ordering asks whether the i-th event comes before the j-th. Answering yes when i < j,
    # and the more common answer when i = j, uses only the question.
    q = questions("test", list(LENGTHS))
    q = q[q.task_type == "ordering"]
    i, j = np.array(
        [list(map(int, re.findall(r"(\d+)(?:st|nd|rd|th)", t)))[:2] for t in q.question]
    ).T
    yes = (q.answer == "yes").to_numpy()
    guess = np.where(i == j, yes[i == j].mean() >= 0.5, i < j)
    lines.append(
        f"Ordering, yes when the first ordinal is smaller: {(guess == yes).mean():.3f} (n {len(q)})"
    )
    for k, r in haystack.items():
        w = cell_weights(r, ["ordering"])
        lines.append(f"Ordering, {k}: {(r['right'] * w).sum():.3f}")
    (OUT / "discussion.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def figure(london: dict, haystack: dict[str, dict]) -> None:
    """London day total's exact match by how many days back the asked day lies, and
    TS-Haystack accuracy by recording length as the mean over the tasks at that length, as in
    the paper's curves. One color per system, dashed for the annotated events."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.4), sharey=True)
    styles = {
        "Memory": (BLUE, "-"),
        "Compression": (AQUA, "-"),
        "No memory": (MUTED, "-"),
    }
    fam, age, household = london["family"], london["age"], london["household"]
    ax = axes[0]
    for name, (color, ls) in styles.items():
        x = london["systems"][name]
        xs, ys, los, his = [], [], [], []
        for lo, hi in BINS:
            sel = (fam == "recall") & (age >= lo) & (age <= hi)
            a, b = bootstrap(x[sel], household[sel])
            xs.append((lo + hi) / 2)
            ys.append(x[sel].mean())
            los.append(a)
            his.append(b)
        ax.plot(xs, ys, color=color, ls=ls, lw=1.5, marker="o", ms=3)
        ax.fill_between(xs, los, his, color=color, alpha=0.12, lw=0)
        print(f"London day total, {name} by days back: " + " ".join(f"{y:.2f}" for y in ys))
    ax.set_title("LCL, only day total", fontsize=8, color=INK)
    ax.set_xlabel("Days back of the asked day", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 830)
    names = {
        "Memory, ECG segments": ("Memory", "-"),
        "Compression, ECG segments": ("Compression", "-"),
        "No memory": ("No memory", "-"),
        "Memory, annotated events": ("Memory", "--"),
        "Compression, annotated events": ("Compression", "--"),
    }
    ax = axes[1]
    xs = np.arange(len(LENGTHS))
    for name, (system, ls) in names.items():
        r = haystack.get(name)
        if r is None:
            continue
        ys = []
        for s in LENGTHS:
            at = r["length"] == s
            ys.append(
                np.mean(
                    [r["right"][at & (r["task"] == t)].mean() for t in np.unique(r["task"][at])]
                )
            )
        ax.plot(xs, ys, color=styles[system][0], ls=ls, lw=1.5, marker="o", ms=3)
        print(f"TS-Haystack {name} by length: " + " ".join(f"{y:.2f}" for y in ys))
    ax.set_xticks(xs, list(LENGTHS.values()))
    ax.set_xlim(-0.25, len(LENGTHS) - 0.75)
    ax.set_title("TS-Haystack, all tasks", fontsize=8, color=INK)
    ax.set_xlabel("Recording length", fontsize=7.5, color=MUTED)
    axes[0].set_ylabel("Accuracy", fontsize=7.5, color=MUTED)
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=3, label="Memory"),
        plt.Line2D([], [], color=AQUA, lw=1.5, marker="o", ms=3, label="Compression"),
        plt.Line2D([], [], color=MUTED, lw=1.5, marker="o", ms=3, label="No memory"),
        plt.Line2D([], [], color=MUTED, lw=1.5, ls="--", label="Annotated events"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "results.pdf")
    print(OUT / "results.pdf")


def seeds(prefix: str) -> list[str]:
    return [f"{prefix}_s{k}" for k in range(3)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    memory = seeds("memory")
    london = london_results(memory, seeds("compression"), seeds("none"), "no_retrieval_s0")
    london_table(london)
    systems = {
        "No memory": (seeds("none"), "none"),
        "Compression, ECG segments": (seeds("tiles_compression"), "tiles"),
        "Memory, ECG segments": (seeds("tiles"), "tiles"),
        "Compression, annotated events": (seeds("events_compression"), "events"),
        "Memory, annotated events": (seeds("events"), "events"),
    }
    haystack = haystack_table(systems)
    discussion(memory, london, haystack)
    figure(london, haystack)


if __name__ == "__main__":
    main()
