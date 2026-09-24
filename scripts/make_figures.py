"""Figure for the proposal: ../proposal/fig_context.pdf, from runs/results.json.

Style follows the Agentic-AutoRAG paper's _figstyle.py (Type-42 fonts, outlined with Ghostscript,
tab10 colours), with Latin Modern to match the proposal's body font. A sidecar records the inputs,
git commit and date.
"""

import datetime
import glob
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

PROPOSAL = Path(__file__).resolve().parents[2] / "proposal"
BLUE, ORANGE, GREEN, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"


def style() -> None:
    for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf"):
        fm.fontManager.addfont(f)
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def outline(path: Path) -> None:
    if shutil.which("gs") is None:
        return
    tmp = path.with_name(path.stem + "__outlined.pdf")
    r = subprocess.run(
        ["gs", "-o", str(tmp), "-sDEVICE=pdfwrite", "-dNoOutputFonts", str(path)],
        capture_output=True,
    )
    if r.returncode == 0:
        tmp.replace(path)


# (label, arm, colour, marker): the fine-tuned reader (circles), the frozen reader read by a
# linear probe (squares) and trees on the same embeddings, all signal only.
ROWS = [
    ("Fine-tuned reader, no history", "S-none", GRAY, "o"),
    ("Fine-tuned reader, own history", "S-own", BLUE, "o"),
    ("Fine-tuned reader, other patient's", "S-swap", ORANGE, "o"),
    ("Frozen reader + linear, no history", "probe-S-none", GRAY, "s"),
    ("Frozen reader + linear, own history", "probe-S-own", BLUE, "s"),
    ("Frozen reader + linear, other patient's", "probe-S-swap", ORANGE, "s"),
    ("Trees, current ECG", "stat:S-current", GREEN, "^"),
    ("Trees, with history", "stat:S-history", GREEN, "^"),
]
# The retrieval results are reported in the text; set True for a second row with them.
RETRIEVAL_PANEL = False
RETRIEVAL = [
    ("Reader, own history", "S-ret-own", BLUE),
    ("+ 4 similar patients with their next ECG", "S-ret-both", BLUE),
    ("+ 4 similar patients, next ECGs shuffled", "S-ret-shuf", ORANGE),
    ("4 similar patients, no own history", "S-ret-nb", GRAY),
    ("Trees with history", "stat:S-history", GREEN),
    ("Trees with history + neighbour vote", "stat:S-history+votes", GREEN),
]


def dots(ax, r: dict, keys: list[str], colours: list[str], markers: list[str]) -> None:
    rows = list(zip(keys, colours, markers, strict=True))
    for y, (key, color, marker) in enumerate(reversed(rows)):
        est, (lo, hi) = r[key], r[f"{key} ci"]
        ax.errorbar(
            est, y, xerr=[[est - lo], [hi - est]], fmt=marker, ms=3.5, color=color, capsize=2, lw=1
        )
    ax.axvline(0.5, color=GRAY, lw=0.6, ls=":")


def main() -> None:
    style()
    r = json.loads(Path("runs/results.json").read_text())
    ret = "auroc S-ret-both" in r and RETRIEVAL_PANEL
    curve = "curve k0" in r
    fig = plt.figure(figsize=(7.0, 3.2 if ret else 1.8))
    grid = fig.add_gridspec(
        2 if ret else 1,
        3 if curve else 2,
        width_ratios=[1, 1, 0.75] if curve else [1, 1],
        height_ratios=[1, 1] if ret else [1],
        hspace=0.7,
        wspace=0.16,
    )
    panels = {f"Test patients ({r['n test']:,})": "auroc"}
    if "confirm auroc probe-S-own" in r:
        panels[f"Fresh patients, confirmation ({r['n confirm']:,})"] = "confirm auroc"
    axes = [fig.add_subplot(grid[0, j]) for j in range(len(panels))]
    for ax, (title, prefix) in zip(axes, panels.items(), strict=True):
        keys = [f"{prefix} {arm}" for _, arm, _, _ in ROWS]
        dots(ax, r, keys, [c for _, _, c, _ in ROWS], [m for _, _, _, m in ROWS])
        ax.axhline(1.5, color=GRAY, lw=0.4, alpha=0.5)
        ax.axhline(4.5, color=GRAY, lw=0.4, alpha=0.5)
        ax.set_yticks(range(len(ROWS)), [lab for lab, _, _, _ in reversed(ROWS)])
        ax.set_title(title, loc="left")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    his = [v[1] for k, v in r.items() if k.endswith(" ci") and not k.startswith("curve")]
    for ax in axes:
        ax.set_xlim(0.55, max(his) + 0.01)
        ax.set_xlabel("AUROC (95 % interval)")
    if curve:
        ax = fig.add_subplot(grid[0, 2])
        ks = [0, 1, 2, 3]
        est = [r[f"curve k{k}"] for k in ks]
        lo = [e - r[f"curve k{k} ci"][0] for k, e in zip(ks, est, strict=True)]
        hi = [r[f"curve k{k} ci"][1] - e for k, e in zip(ks, est, strict=True)]
        ax.errorbar(ks, est, yerr=[lo, hi], fmt="o-", ms=3.5, color=BLUE, capsize=2, lw=1)
        ax.set_xticks(ks)
        ax.set_xlabel("Own earlier ECGs given")
        ax.set_title("Fine-tuned, AUROC by history", loc="left")
    if ret:
        ax = fig.add_subplot(grid[1, :])
        keys = [f"auroc {a}" for _, a, _ in RETRIEVAL]
        dots(ax, r, keys, [c for _, _, c in RETRIEVAL], ["o"] * len(RETRIEVAL))
        ax.set_yticks(range(len(RETRIEVAL)), [lab for lab, _, _ in reversed(RETRIEVAL)])
        ax.set_title("Retrieval on top of memory, signal only", loc="left")
        ax.set_xlim(0.48, max(his) + 0.01)
        ax.set_xlabel("AUROC (95 % interval)")
    out = PROPOSAL / "fig_context.pdf"
    fig.savefig(out)
    outline(out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    sidecar = {
        "inputs": ["runs/results.json"],
        "commit": commit.strip(),
        "date": datetime.date.today().isoformat(),
    }
    Path("runs/fig_context.sidecar.json").write_text(json.dumps(sidecar, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
