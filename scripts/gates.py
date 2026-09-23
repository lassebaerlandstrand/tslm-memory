"""Write gates.json: the gates and thresholds, fixed before any trained reader is scored on test.

tau (the least gain from own history that counts) = min(0.02, half the matched signal
statistician's history gain on test). Needs runs/baselines/test/S-*.jsonl. Commit gates.json
before running evaluate.py on test.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tslm_context.grade import auroc_within


def auroc(name: str) -> float:
    d = pd.read_json(f"runs/baselines/test/{name}.jsonl", lines=True)
    return auroc_within(d.y.to_numpy(), d.p_yes.to_numpy(), d.stratum.to_numpy())


def main() -> None:
    gain = auroc("S-history") - auroc("S-current")
    assert gain > 0, f"signal statistician history gain {gain:.4f}: set tau by hand"
    tau = float(min(0.02, gain / 2))
    gates = {
        "metric": "mean AUROC within the 8 strata (finding x current state), seed-averaged p_yes",
        "interval": "95 %, SE_tot^2 = SE_boot^2 + Var(per-seed contrasts)/n_seeds; bootstrap "
        "within the 16 cells, B = 2000, seed 0",
        "order": ["P1", "P2", "G1", "G2", "G3"],
        "P1": "stage 1 passes in every seed (runs/stage1_s*.json)",
        "P2": "on validation: S-none >= 0.60 and S-none minus S-none-qswap >= 0.05",
        "G1": "S-own minus S-swap: interval above 0 and every per-seed contrast above 0",
        "G2": "S-own minus S-none >= tau, interval above 0, every per-seed contrast above 0, and "
        "L = (S-own - S-swap) - 2/3 (S-own - S-none) >= 0",
        "G3": "5-fold cross-validated logistic stack per finding (folds stratified by cell, "
        "current state as covariate) of T-history plus logit(S-own), minus T-history alone: "
        "bootstrap interval above 0",
        "tau": tau,
        "signal statistician history gain on test": float(gain),
    }
    Path("gates.json").write_text(json.dumps(gates, indent=1) + "\n")
    print(json.dumps({"tau": tau, "gain": round(gain, 4)}))


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
