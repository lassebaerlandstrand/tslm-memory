"""Evaluate the gates in gates.json and write runs/results.json and the LaTeX macros for the
proposal (../proposal/tslm-context-results.tex). Prints aggregates only."""

import itertools
import json
import re
import subprocess
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from tslm_context.grade import auroc_within, bootstrap, ci
from tslm_context.samples import Queries

OUT = Path(__file__).resolve().parents[2] / "proposal" / "tslm-context-results.tex"
SEEDS = [0, 1, 2]


def load(role: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Per arm: p_yes [n, seeds] in seed order; plus the shared query frame (y, stratum, cell)."""
    arms, base = {}, None
    for path in sorted(glob(f"runs/eval/{role}/*_s*.jsonl")):
        m = re.fullmatch(r"(.+)_s(\d+)\.jsonl", Path(path).name)
        d = (
            pd.read_json(path, lines=True, dtype={"cell": str})
            .sort_values("i")
            .reset_index(drop=True)
        )
        if base is None:
            base = d[["i", "finding", "cell", "stratum", "y", "n_hist"]]
        assert (d.i.to_numpy() == base.i.to_numpy()).all()
        arms.setdefault(m[1], {})[int(m[2])] = d.p_yes.to_numpy()
    for arm, by_seed in arms.items():
        assert sorted(by_seed) == SEEDS, (role, arm, sorted(by_seed))
    out = {k: np.stack([v[s] for s in SEEDS], axis=1) for k, v in arms.items()}
    for path in sorted(glob(f"runs/baselines/{role}/*.jsonl")):
        d = pd.read_json(path, lines=True).sort_values("i").reset_index(drop=True)
        assert (d.i.to_numpy() == base.i.to_numpy()).all()
        out["stat:" + Path(path).stem] = d.p_yes.to_numpy()[:, None]
    return out, base


class Scores:
    def __init__(self, arms: dict[str, np.ndarray], base: pd.DataFrame):
        self.arms = arms
        self.y, self.strata = base.y.to_numpy(), base.stratum.to_numpy()
        self.cells = (base.finding + "|" + base.cell).to_numpy()

    def auc(self, arm: str, idx=slice(None), seed: int | None = None) -> float:
        p = self.arms[arm]
        p = p.mean(axis=1) if seed is None else p[:, seed]
        return auroc_within(self.y[idx], p[idx], self.strata[idx])

    def contrast(self, a: str, b: str, sub: np.ndarray | None = None) -> dict:
        idx = np.arange(len(self.y)) if sub is None else np.flatnonzero(sub)
        est = self.auc(a, idx) - self.auc(b, idx)
        n = min(self.arms[a].shape[1], self.arms[b].shape[1])
        per_seed = (
            [self.auc(a, idx, s) - self.auc(b, idx, s) for s in range(n)]
            if self.arms[a].shape[1] == self.arms[b].shape[1] > 1
            else []
        )
        boots = bootstrap(lambda i: self.auc(a, idx[i]) - self.auc(b, idx[i]), self.cells[idx])
        lo, hi = ci(est, boots, per_seed)
        return {"est": est, "lo": lo, "hi": hi, "per_seed": per_seed}


def stack(sc: Scores, base: pd.DataFrame) -> dict:
    """G3: does the reader reading earlier ECGs add to the full machine read with history?"""

    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    t = logit(sc.arms["stat:T-history"][:, 0])
    s = logit(sc.arms["S-own"].mean(axis=1))
    # y does not depend on the current state by design (balanced cells), so without the state as
    # a covariate a reader's state offset would look like noise and shrink its weight.
    now = base.cell.str[0].astype(float).to_numpy()
    both, alone = np.zeros(len(t)), np.zeros(len(t))
    for f in base.finding.unique():
        m = np.flatnonzero((base.finding == f).to_numpy())
        folds = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in folds.split(m, base.cell.to_numpy()[m]):
            for x, out in ((np.c_[t, s, now], both), (np.c_[t, now], alone)):
                clf = LogisticRegression().fit(x[m[tr]], sc.y[m[tr]])
                out[m[te]] = clf.decision_function(x[m[te]])
    sc.arms["stack:T-history+S-own"], sc.arms["stack:T-history"] = both[:, None], alone[:, None]
    return sc.contrast("stack:T-history+S-own", "stack:T-history")


def label_silent(base: pd.DataFrame) -> np.ndarray:
    """Queries whose earlier ECGs never carry the finding asked about."""
    q = Queries("test")
    assert (q.q.index.to_numpy() == base.i.to_numpy()).all()
    return np.array(
        [not any(r.finding in q.ecgs.loc[s, "labels"] for s in r.earlier) for r in q.q.itertuples()]
    )


def macro(key: str) -> str:
    words = re.sub(r"[^A-Za-z0-9]+", " ", key).title().replace(" ", "")
    return "ctx" + re.sub(
        r"\d",
        lambda m: "zero one two three four five six seven eight nine".split()[
            int(m.group())
        ].title(),
        words,
    )


def main() -> None:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "gates.json"], check=False)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "gates.json"], check=False)
    assert tracked.returncode == 0 and clean.returncode == 0, "gates.json must be committed as is"
    gates = json.loads(Path("gates.json").read_text())
    res = {}
    arms, base = load("val")
    val = Scores(arms, base)
    stage1 = [json.loads(Path(f"runs/stage1_s{k}.json").read_text()) for k in SEEDS]
    res["P1"] = all(r["pass"] for r in stage1)
    res["val S-none"] = val.auc("S-none")
    res["val S-none minus qswap"] = val.auc("S-none") - val.auc("S-none-qswap")
    res["P2"] = res["val S-none"] >= 0.60 and res["val S-none minus qswap"] >= 0.05

    arms, base = load("test")
    sc = Scores(arms, base)
    for arm in sorted(sc.arms):
        res[f"auroc {arm}"] = sc.auc(arm)
        n = sc.arms[arm].shape[1]
        per_seed = [sc.auc(arm, seed=k) for k in range(n)] if n > 1 else []
        boots = bootstrap(lambda i, a=arm: sc.auc(a, i), sc.cells, n=1000)
        res[f"auroc {arm} ci"] = ci(res[f"auroc {arm}"], boots, per_seed)
    g1 = sc.contrast("S-own", "S-swap")
    g2 = sc.contrast("S-own", "S-none")
    res["S-own minus S-swap"], res["S-own minus S-none"] = g1, g2
    res["G1"] = g1["lo"] > 0 and len(g1["per_seed"]) == 3 and all(x > 0 for x in g1["per_seed"])
    res["L"] = g1["est"] - 2 / 3 * g2["est"]
    res["G2"] = (
        g2["est"] >= gates["tau"]
        and g2["lo"] > 0
        and len(g2["per_seed"]) == 3
        and all(x > 0 for x in g2["per_seed"])
        and res["L"] >= 0
    )
    res["G3 stack"] = stack(sc, base)
    res["G3"] = res["G3 stack"]["lo"] > 0
    # Reported, not gated.
    for a, b in [
        ("T-own", "T-swap"),
        ("T-own", "T-none"),
        ("S-k0", "S-none"),
        ("T-k0", "T-none"),
        ("T-own", "S-own"),
        ("stat:S-history", "stat:S-current"),
        ("stat:T-history", "stat:T-current"),
        ("stat:T-history-42k", "stat:T-current-42k"),
    ]:
        res[f"{a} minus {b}"] = sc.contrast(a, b)
    silent = label_silent(base)
    res["label-silent share"] = float(silent.mean())
    res["label-silent S-own minus S-swap"] = sc.contrast("S-own", "S-swap", silent)
    gain = res["stat:S-history minus stat:S-current"]["est"]
    if gain > 0:
        res["reader gain as share of statistician gain"] = g2["est"] / gain
    # Gates are read in order: a later gate counts only if every earlier one passed.
    res["gates passed"] = list(itertools.takewhile(lambda g: res[g], gates["order"]))
    res["tau"] = gates["tau"]
    res["n test"], res["n val"], res["n train"] = len(sc.y), len(val.y), len(Queries("train"))
    sel = json.loads(Path("runs/selection.json").read_text())
    res["n pairs"], res["n mimic ecgs"] = sel["n pairs"], sel["n ecgs"]
    enc = json.loads(Path("runs/encoder_check.json").read_text())
    res["encoder spread"] = enc["encoder output spread across tokens"]
    res["encoder magnitude"] = enc["encoder output magnitude"]
    res["encoder soft tokens"] = enc["n_soft_tokens"]
    res["encoder distinct tokens"] = enc["distinct bf16 soft tokens"]
    Path("runs/results.json").write_text(json.dumps(res, indent=1, default=float))

    lines = ["% Generated by tslm-context/scripts/make_results.py. Do not edit."]
    for k, v in res.items():
        if isinstance(v, bool):
            lines.append(f"\\newcommand{{\\{macro(k)}}}{{{'passed' if v else 'failed'}}}")
        elif isinstance(v, int):
            lines.append(f"\\newcommand{{\\{macro(k)}}}{{{v:,}}}".replace(",", "{,}"))
        elif isinstance(v, float):
            lines.append(f"\\newcommand{{\\{macro(k)}}}{{{fmt(v)}}}")
        elif isinstance(v, dict):
            for part in ("est", "lo", "hi"):
                lines.append(
                    f"\\newcommand{{\\{macro(k + ' ' + part)}}}{{{fmt(v[part], sign=True)}}}"
                )
    OUT.write_text("\n".join(lines) + "\n")
    for k, v in res.items():
        print(
            k,
            {a: round(b, 3) for a, b in v.items() if a != "per_seed"}
            if isinstance(v, dict)
            else (round(v, 3) if isinstance(v, float) else v),
        )


def fmt(x: float, sign: bool = False) -> str:
    if not sign and x != 0 and abs(x) < 1e-3:
        mant, exp = f"{x:.1e}".split("e")
        return f"${mant}\\times 10^{{{int(exp)}}}$"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    s = s.replace("-0.000", "0.000").replace("+0.000", "0.000")
    return s.replace("-", "\\textminus ")


if __name__ == "__main__":
    main()
