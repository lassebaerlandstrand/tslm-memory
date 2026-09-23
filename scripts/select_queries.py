"""Select the readout-B queries (see tslm_context.selection) and list the records to fetch.

Writes data/ecgs.parquet (canonical table), data/queries.parquet and data/fetch.txt. Prints counts
only.
"""

import numpy as np

from tslm_context.mimic import canonical_table
from tslm_context.selection import select, to_fetch
from tslm_context.waves import DATA

PER_CELL = {"test": 200, "val": 25, "train": 500}


def main() -> None:
    ecgs = canonical_table()
    ecgs.to_parquet(DATA / "ecgs.parquet")
    q = select(ecgs, PER_CELL)
    assert q.subject_id.is_unique
    q.to_parquet(DATA / "queries.parquet")
    studies = to_fetch(q)
    assert not set(q.next_study) & set(studies)
    path = dict(zip(ecgs.study_id, ecgs.path, strict=True))
    files = [f"{path[s]}.{ext}" for s in studies for ext in ("hea", "dat")]
    (DATA / "fetch.txt").write_text("\n".join(files) + "\n")
    for role in ("test", "val", "train"):
        r = q[q.role == role]
        print(
            f"{role}: queries {len(r)}, subjects {r.subject_id.nunique()}, "
            f"with 3 earlier ECGs {(r.n_hist == 3).mean():.2f}, "
            f"median gap {r.gap_days.median():.1f} d, "
            f"median days since last earlier ECG {np.nanmedian(r.hist_days_0):.1f}"
        )
    print(f"records to fetch {len(studies)}, files {len(files)}")


if __name__ == "__main__":
    main()
