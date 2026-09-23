"""ECGFounder embeddings for every fetched record (query and own history of every query).

Writes data/emb.npy [N, 1024] float32 and data/emb.parquet (study_id, row, unusable). Unusable
records (a lead missing throughout, or every lead flat) get no embedding row. Prints counts only.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from tslm_context import encode
from tslm_context.waves import DATA, read500, unusable


def main() -> None:
    q = pd.read_parquet(DATA / "queries.parquet")
    cols = ["cur_study"] + [f"hist_study_{j}" for j in range(3)]
    studies = np.unique(q[cols].to_numpy().ravel())
    studies = studies[studies >= 0]
    ecgs = pd.read_parquet(DATA / "ecgs.parquet", columns=["study_id", "path"])
    path = dict(zip(ecgs.study_id, ecgs.path, strict=True))
    model = encode.ecgfounder()
    emb, bad, n_nan = [], [], 0
    with ThreadPoolExecutor(16) as ex:
        for s in range(0, len(studies), 512):
            chunk = studies[s : s + 512]
            sigs = list(ex.map(lambda i: read500(path[i]), chunk))
            flags = [unusable(x) for x in sigs]
            n_nan += sum(np.isnan(x).any() for x, f in zip(sigs, flags, strict=True) if not f)
            keep = [np.nan_to_num(x) for x, f in zip(sigs, flags, strict=True) if not f]
            if keep:
                emb.append(encode.embed(model, np.stack(keep)))
            bad += flags
    emb = np.concatenate(emb)
    bad = np.array(bad)
    rows = np.full(len(studies), -1)
    rows[~bad] = np.arange(len(emb))
    pd.DataFrame({"study_id": studies, "row": rows, "unusable": bad}).to_parquet(
        DATA / "emb.parquet"
    )
    np.save(DATA / "emb.npy", emb)
    print(
        f"records {len(studies)}, unusable {int(bad.sum())}, with short dropouts set to 0 {n_nan}, "
        f"embeddings {emb.shape}"
    )


if __name__ == "__main__":
    main()
