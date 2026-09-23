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


PANELS = {
    "Signal only (no machine read)": [
        ("Reader, never saw history", "S-none", GRAY),
        ("Reader with history, none given", "S-k0", GRAY),
        ("Reader, own 3 earlier ECGs", "S-own", BLUE),
        ("Reader, another patient's ECGs", "S-swap", ORANGE),
        ("Trees on embedding, current", "stat:S-current", GREEN),
        ("Trees on embedding, with history", "stat:S-history", GREEN),
    ],
    "Text (machine read of every ECG)": [
        ("Reader, never saw history", "T-none", GRAY),
        ("Reader with history, none given", "T-k0", GRAY),
        ("Reader, own 3 earlier ECGs", "T-own", BLUE),
        ("Reader, another patient's ECGs", "T-swap", ORANGE),
        ("Trees on machine read, current", "stat:T-current", GREEN),
        ("Trees on machine read, with history", "stat:T-history", GREEN),
    ],
}


def main() -> None:
    style()
    r = json.loads(Path("runs/results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), sharex=True, gridspec_kw={"wspace": 1.1})
    for ax, (title, rows) in zip(axes, PANELS.items(), strict=True):
        for y, (_label, arm, color) in enumerate(reversed(rows)):
            est = r[f"auroc {arm}"]
            lo, hi = r[f"auroc {arm} ci"]
            ax.errorbar(
                est, y, xerr=[[est - lo], [hi - est]], fmt="o", ms=3.5, color=color, capsize=2, lw=1
            )
        ax.set_yticks(range(len(rows)), [lab for lab, _, _ in reversed(rows)])
        ax.axvline(0.5, color=GRAY, lw=0.6, ls=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUROC within finding and current state")
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
