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


ROWS = [
    "Reader never trained with history",
    "Reader trained with history, none given",
    "Reader, own 3 earlier ECGs",
    "Reader, another patient's earlier ECGs",
    "Gradient-boosted trees, current ECG",
    "Gradient-boosted trees, with history",
]
MEMORY = {
    "Memory, signal only": [
        "S-none",
        "S-k0",
        "S-own",
        "S-swap",
        "stat:S-current",
        "stat:S-history",
    ],
    "Memory, with the machine read": [
        "T-none",
        "T-k0",
        "T-own",
        "T-swap",
        "stat:T-current",
        "stat:T-history",
    ],
}
COLOURS = [GRAY, GRAY, BLUE, ORANGE, GREEN, GREEN]
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


def dots(ax, r: dict, arms: list[str], colours: list[str]) -> None:
    for y, (arm, color) in enumerate(zip(reversed(arms), reversed(colours), strict=True)):
        est = r[f"auroc {arm}"]
        lo, hi = r[f"auroc {arm} ci"]
        ax.errorbar(
            est, y, xerr=[[est - lo], [hi - est]], fmt="o", ms=3.5, color=color, capsize=2, lw=1
        )
    ax.axvline(0.5, color=GRAY, lw=0.6, ls=":")


def main() -> None:
    style()
    r = json.loads(Path("runs/results.json").read_text())
    ret = "auroc S-ret-both" in r and RETRIEVAL_PANEL
    curve = "curve k0" in r
    fig = plt.figure(figsize=(7.0, 3.1 if ret else 1.7))
    grid = fig.add_gridspec(
        2 if ret else 1,
        3 if curve else 2,
        width_ratios=[1, 1, 0.75] if curve else [1, 1],
        height_ratios=[1, 1] if ret else [1],
        hspace=0.7,
        wspace=0.12,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax, (title, arms) in zip(axes, MEMORY.items(), strict=True):
        dots(ax, r, arms, COLOURS)
        ax.set_yticks(range(len(ROWS)), list(reversed(ROWS)))
        ax.set_title(title, loc="left")
    axes[1].set_yticklabels([])
    his = [v[1] for k, v in r.items() if k.endswith(" ci") and not k.startswith("curve")]
    for ax in axes:
        ax.set_xlim(0.48, max(his) + 0.01)
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
        ax.set_ylabel("AUROC", labelpad=1)
        ax.set_title("More history, same patients", loc="left")
    if ret:
        ax = fig.add_subplot(grid[1, :])
        dots(ax, r, [a for _, a, _ in RETRIEVAL], [c for _, _, c in RETRIEVAL])
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
