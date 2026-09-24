"""Evaluate the confirmation gates (gates_confirm.json) on the confirmation set and write
runs/results_confirm.json, with the same scores on test for reference. C2 is read only if C1
passes. Prints aggregates only.

Also reported, not gated: probe-S-dates (scripts/probe.py), added after the confirmation was
scored. The own-history prompt shows how many earlier ECGs there are and how long before now each
was taken, and that alone can predict the answer. S-dates keeps those lines and drops the
recordings, so own minus dates is what the earlier recordings add beyond their dates."""

import json
from pathlib import Path

from make_results import Scores, load

from tslm_context.grade import bootstrap, ci

ARMS = ["probe-S-none", "probe-S-own", "probe-S-swap", "probe-S-dates", "S-none", "S-own"]
ARMS += ["S-swap", "stat:S-current", "stat:S-history"]
C1, C2 = "probe-S-own minus probe-S-swap", "probe-S-own minus probe-S-none"
CONTRASTS = {
    C1: ("probe-S-own", "probe-S-swap"),
    C2: ("probe-S-own", "probe-S-none"),
    "probe-S-own minus S-own": ("probe-S-own", "S-own"),
    "probe-S-none minus S-none": ("probe-S-none", "S-none"),
    "S-own minus S-none": ("S-own", "S-none"),
    "probe-S-own minus stat:S-history": ("probe-S-own", "stat:S-history"),
    "probe-S-own minus probe-S-dates": ("probe-S-own", "probe-S-dates"),
    "probe-S-dates minus probe-S-none": ("probe-S-dates", "probe-S-none"),
}


def scores(role: str) -> dict:
    arms, base = load(role)
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
