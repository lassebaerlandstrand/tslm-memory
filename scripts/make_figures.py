"""Figure for the proposal: ../proposal/fig_memory.pdf, from runs/results.json.

The frozen reader on the fresh confirmation patients: without history, with only the dates of the
patient's earlier ECGs, with the earlier ECGs themselves, and with another patient's. Style
follows the Agentic-AutoRAG paper's _figstyle.py (Type-42 fonts outlined with Ghostscript) with
the Okabe-Ito colours of the TikZ figure, in TeX Gyre Termes to match the ACL template's Times. A
sidecar records the inputs, git commit and date.
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
BLUE, SKY, ORANGE, GRAY = "#0072B2", "#56B4E9", "#E69F00", "#999999"
BARS = [
    ("No history", "probe-S-none", GRAY),
    ("Only the dates of\nown earlier ECGs", "probe-S-dates", SKY),
    ("Own earlier ECGs", "probe-S-own", BLUE),
    ("Another patient\u2019s\nearlier ECGs", "probe-S-swap", ORANGE),
]


def style() -> None:
    for f in glob.glob("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"):
        fm.fontManager.addfont(f)
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["TeX Gyre Termes", "STIX Two Text", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
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


def main() -> None:
    style()
    r = json.loads(Path("runs/results.json").read_text())
    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    for y, (_, arm, colour) in enumerate(reversed(BARS)):
        est, (lo, hi) = r[f"confirm auroc {arm}"], r[f"confirm auroc {arm} ci"]
        # Bars start at 0.5, where AUROC is chance.
        ax.barh(y, est - 0.5, left=0.5, height=0.62, color=colour)
        ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], color="black", lw=0.8, capsize=2)
        ax.text(hi + 0.004, y, f"{est:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(BARS)), [label for label, _, _ in reversed(BARS)])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, 0.8)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("AUROC, next ECG (0.5 = chance)")
    out = PROPOSAL / "fig_memory.pdf"
    fig.savefig(out)
    outline(out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    sidecar = {
        "inputs": ["runs/results.json"],
        "commit": commit.strip(),
        "date": datetime.date.today().isoformat(),
    }
    Path("runs/fig_memory.sidecar.json").write_text(json.dumps(sidecar, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
