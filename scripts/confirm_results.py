"""Evaluate the confirmation gates (gates_confirm.json) on the confirmation set and write
runs/results_confirm.json, with the same scores on test for reference. C2 is read only if C1
passes. Prints aggregates only.

Also reported, not gated: a dates-only arm. The own-history prompt shows how many earlier ECGs
there are and how long before now each was taken, and that alone predicts the answer. The arm
combines the no-history probe's score with this count and these gaps, without the recordings,
by a logistic regression per stratum fitted on the other role (test for confirm and back)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from make_results import Scores, load
from sklearn.linear_model import LogisticRegression

from tslm_context.grade import bootstrap, ci
from tslm_context.samples import Queries
from tslm_context.waves import DATA

ARMS = ["probe-S-none", "probe-S-own", "probe-S-swap", "S-none", "S-own", "S-swap"]
ARMS += ["stat:S-current", "stat:S-history", "dates:probe-S-none"]
C1, C2 = "probe-S-own minus probe-S-swap", "probe-S-own minus probe-S-none"
CONTRASTS = {
    C1: ("probe-S-own", "probe-S-swap"),
    C2: ("probe-S-own", "probe-S-none"),
    "probe-S-own minus S-own": ("probe-S-own", "S-own"),
    "probe-S-none minus S-none": ("probe-S-none", "S-none"),
    "S-own minus S-none": ("S-own", "S-none"),
    "probe-S-own minus stat:S-history": ("probe-S-own", "stat:S-history"),
    "probe-S-own minus dates:probe-S-none": ("probe-S-own", "dates:probe-S-none"),
    "dates:probe-S-none minus probe-S-none": ("dates:probe-S-none", "probe-S-none"),
}


def dates(role: str, base: pd.DataFrame) -> np.ndarray:
    """Count of earlier ECGs and log days before now per slot, aligned with base.i."""
    frames = (
        pd.read_parquet(DATA / "queries.parquet"),
        pd.read_parquet(DATA / "ecgs.parquet", columns=["study_id"]).set_index("study_id"),
        pd.read_parquet(DATA / "emb.parquet").set_index("study_id").row,
    )
    q = Queries(role, frames).q.iloc[base.i.to_numpy()]
    days = np.array([(d + [np.nan] * 3)[:3] for d in q.days], dtype=float)
    return np.c_[np.isnan(days[:, 2]), np.log1p(np.nan_to_num(days))]


def dates_arm(fit: tuple, score: tuple) -> np.ndarray:
    """Per seed, the no-history logit plus dates, fitted per stratum on `fit`, scored on `score`."""
    (fa, fb, fx), (sa, sb, sx) = fit, score
    out = np.zeros(sa["probe-S-none"].shape)
    for s in range(out.shape[1]):
        for k in np.unique(sb.stratum):
            f, m = (fb.stratum == k).to_numpy(), (sb.stratum == k).to_numpy()
            p, ps = fa["probe-S-none"][f, s], sa["probe-S-none"][m, s]
            lr = LogisticRegression(max_iter=2000).fit(
                np.c_[np.log(p / (1 - p)), fx[f]], fb.y.to_numpy()[f]
            )
            out[m, s] = lr.predict_proba(np.c_[np.log(ps / (1 - ps)), sx[m]])[:, 1]
    return out


def scores(arms: dict, base: pd.DataFrame) -> dict:
    sc = Scores(arms, base)
    out = {}
    for arm in ARMS:
        est = sc.auc(arm)
        per_seed = [sc.auc(arm, seed=s) for s in range(arms[arm].shape[1])]
        lo, hi = ci(est, bootstrap(lambda i, a=arm: sc.auc(a, i), sc.cells), per_seed)
        out[f"auroc {arm}"] = {"est": est, "lo": lo, "hi": hi}
    for name, (a, b) in CONTRASTS.items():
        out[name] = sc.contrast(a, b)
    return out


def main() -> None:
    tau = json.loads(Path("gates_confirm.json").read_text())["tau"]
    data = {}
    for role in ("confirm", "test"):
        arms, base = load(role)
        data[role] = (arms, base, dates(role, base))
    for role, other in (("confirm", "test"), ("test", "confirm")):
        data[role][0]["dates:probe-S-none"] = dates_arm(data[other], data[role])
    res = {role: scores(data[role][0], data[role][1]) for role in ("confirm", "test")}
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
