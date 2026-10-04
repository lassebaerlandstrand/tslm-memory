"""Select the readout-B queries (see tslm_memory_poc.selection) and list the records to fetch.

The confirmation role is drawn afterwards from bucket 1 subjects no other role uses, so the other
roles are the same with or without it.

Writes data/ecgs.parquet (canonical table), data/queries.parquet and data/fetch.txt. Prints counts
only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tslm_memory_poc.mimic import canonical_table
from tslm_memory_poc.selection import CONFIRM, select, to_fetch
from tslm_memory_poc.waves import DATA

PER_CELL = {"test": 200, "val": 25, "train": 500}
CONFIRM_PER_CELL = {"confirm": 200}


def main() -> None:
    ecgs = canonical_table()
    ecgs.to_parquet(DATA / "ecgs.parquet")
    q = select(ecgs, PER_CELL)
    n_pairs = int(q.attrs["n_pairs"])
    confirm = select(ecgs, CONFIRM_PER_CELL, buckets=CONFIRM, exclude=frozenset(q.subject_id))
    q = pd.concat([q, confirm], ignore_index=True)
    assert q.subject_id.is_unique
    q.to_parquet(DATA / "queries.parquet")
    studies = to_fetch(q)
    assert not set(q.next_study) & set(studies)
    path = dict(zip(ecgs.study_id, ecgs.path, strict=True))
    files = [f"{path[s]}.{ext}" for s in studies for ext in ("hea", "dat")]
    (DATA / "fetch.txt").write_text("\n".join(files) + "\n")
    for role in ("test", "val", "train", "confirm"):
        r = q[q.role == role]
        print(
            f"{role}: queries {len(r)}, subjects {r.subject_id.nunique()}, "
            f"with 3 earlier ECGs {(r.n_hist == 3).mean():.2f}, "
            f"median gap {r.gap_days.median():.1f} d, "
            f"median days since last earlier ECG {np.nanmedian(r.hist_days_0):.1f}"
        )
    print(f"records to fetch {len(studies)}, files {len(files)}")
    Path("runs").mkdir(exist_ok=True)
    counts = {"n ecgs": len(ecgs), "n pairs": n_pairs, "n records": len(studies)}
    Path("runs/selection.json").write_text(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
