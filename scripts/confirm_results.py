"""Evaluate the confirmation gates (gates_confirm.json) on the confirmation set and write
runs/results_confirm.json, with the same scores on test for reference. C2 is read only if C1
passes. Prints aggregates only.

Also reported, not gated: probe-S-dates (scripts/probe.py), added after the confirmation was
scored. The own-history prompt shows how many earlier ECGs there are and how long before now each
was taken, and that alone can predict the answer. S-dates keeps those lines and drops the
recordings, so own minus dates is what the earlier recordings add beyond their dates."""

import json
import re
import subprocess
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tslm_memory_poc.grade import auroc_within, bootstrap, ci

SEEDS = [0, 1, 2]
ARMS = ["probe-S-none", "probe-S-dates", "probe-S-own", "probe-S-swap"]
C1, C2 = "probe-S-own minus probe-S-swap", "probe-S-own minus probe-S-none"
CONTRASTS = {
    C1: ("probe-S-own", "probe-S-swap"),
    C2: ("probe-S-own", "probe-S-none"),
    "probe-S-own minus probe-S-dates": ("probe-S-own", "probe-S-dates"),
    "probe-S-dates minus probe-S-none": ("probe-S-dates", "probe-S-none"),
}


def committed(path: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], check=False)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], check=False)
    return tracked.returncode == 0 and clean.returncode == 0


def load(role: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Per arm: p_yes [n, seeds] in seed order; plus the shared query frame (y, stratum, cell)."""
    arms, base = {}, None
    for path in sorted(glob(f"runs/eval/{role}/probe-*_s*.jsonl")):
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
    return {k: np.stack([v[s] for s in SEEDS], axis=1) for k, v in arms.items()}, base


class Scores:
    def __init__(self, arms: dict[str, np.ndarray], base: pd.DataFrame):
        self.arms = arms
        self.y, self.strata = base.y.to_numpy(), base.stratum.to_numpy()
        self.cells = (base.finding + "|" + base.cell).to_numpy()

    def auc(self, arm: str, idx=slice(None), seed: int | None = None) -> float:
        p = self.arms[arm]
        p = p.mean(axis=1) if seed is None else p[:, seed]
        return auroc_within(self.y[idx], p[idx], self.strata[idx])

    def contrast(self, a: str, b: str) -> dict:
        idx = np.arange(len(self.y))
        est = self.auc(a) - self.auc(b)
        per_seed = [self.auc(a, seed=s) - self.auc(b, seed=s) for s in range(len(SEEDS))]
        boots = bootstrap(lambda i: self.auc(a, idx[i]) - self.auc(b, idx[i]), self.cells)
        lo, hi = ci(est, boots, per_seed)
        return {"est": est, "lo": lo, "hi": hi, "per_seed": per_seed}


def scores(role: str) -> dict:
    arms, base = load(role)
    sc = Scores(arms, base)
    out = {}
    for arm in ARMS:
        est = sc.auc(arm)
        per_seed = [sc.auc(arm, seed=s) for s in range(len(SEEDS))]
        lo, hi = ci(est, bootstrap(lambda i, a=arm: sc.auc(a, i), sc.cells), per_seed)
        out[f"auroc {arm}"] = {"est": est, "lo": lo, "hi": hi}
    for name, (a, b) in CONTRASTS.items():
        out[name] = sc.contrast(a, b)
    return out


def main() -> None:
    tau = json.loads(Path("gates_confirm.json").read_text())["tau"]
    res = {role: scores(role) for role in ("confirm", "test")}
    c1, c2 = res["confirm"][C1], res["confirm"][C2]
    res["C1"] = bool(c1["lo"] > 0 and min(c1["per_seed"]) > 0)
    big_l = c1["est"] - 2 / 3 * c2["est"]
    res["L"] = big_l
    res["C2"] = (
        bool(c2["est"] >= tau and c2["lo"] > 0 and min(c2["per_seed"]) > 0 and big_l >= 0)
        if res["C1"]
        else None
    )
    Path("runs/results_confirm.json").write_text(json.dumps(res, indent=1))
    for role in ("confirm", "test"):
        for k, v in res[role].items():
            print(f"{role} {k}: {v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]", end="")
            print(f" per seed {[round(x, 3) for x in v['per_seed']]}" if v.get("per_seed") else "")
    print(f"C1 {res['C1']}, C2 {res['C2']} (tau {tau}, L {big_l:.3f})")


if __name__ == "__main__":
    main()
