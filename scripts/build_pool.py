"""The retrieval pool: training subjects' query ECGs stored with what followed them (their next
ECG), and each query's most similar pool entries.

1. Extracts the pool's next ECGs from the Kaggle copy (data/kaggle/*.zip) with PhysioNet's
   checksums. Only training subjects are in the pool; no test or validation next ECG is read.
2. Embeds them with ECGFounder into data/emb_next.npy and data/emb_next.parquet.
3. For every role, the K most similar pool entries by cosine similarity of the ECGFounder
   embeddings of the query ECGs, never the query's own subject: data/neighbours_{role}.npz.
Usage: build_pool.py {extract,embed}
"""

import hashlib
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from tslm_context.waves import DATA, read500, unusable

K = 8
ZIP = DATA / "kaggle" / "mimic-iv-ecg-demo-diagnostic-ecg-matched-subset.zip"
PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"


def pool_next() -> pd.DataFrame:
    q = pd.read_parquet(DATA / "queries.parquet")
    train, other = q[q.role == "train"], q[q.role != "train"]
    assert not set(train.subject_id) & set(other.subject_id)
    return train


def extract() -> None:
    train = pool_next()
    ecgs = pd.read_parquet(DATA / "ecgs.parquet", columns=["study_id", "path"])
    path = dict(zip(ecgs.study_id, ecgs.path, strict=True))
    sums = dict(
        reversed(line.split(maxsplit=1))
        for line in (DATA / "mimic" / "SHA256SUMS.txt").read_text().splitlines()
    )
    files = [f"{path[s]}.{e}" for s in train.next_study for e in ("hea", "dat")]
    todo = [f for f in files if not (DATA / "mimic" / f).exists()]
    with zipfile.ZipFile(ZIP) as z:
        for f in todo:
            data = z.read(PREFIX + f)
            assert hashlib.sha256(data).hexdigest() == sums[f], "checksum mismatch"
            out = DATA / "mimic" / f
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    print(f"pool next ECGs {len(train)}, files {len(files)}, extracted {len(todo)}")


def embed() -> None:
    import torch

    from tslm_context import encode

    train = pool_next()
    ecgs = pd.read_parquet(DATA / "ecgs.parquet", columns=["study_id", "path"])
    path = dict(zip(ecgs.study_id, ecgs.path, strict=True))
    studies = train.next_study.to_numpy()
    model = encode.ecgfounder()
    emb, bad = [], []
    with ThreadPoolExecutor(16) as ex:
        for s in range(0, len(studies), 512):
            sigs = list(ex.map(lambda i: read500(path[i]), studies[s : s + 512]))
            flags = [unusable(x) for x in sigs]
            keep = [np.nan_to_num(x) for x, f in zip(sigs, flags, strict=True) if not f]
            emb.append(encode.embed(model, np.stack(keep)))
            bad += flags
    emb, bad = np.concatenate(emb), np.array(bad)
    rows = np.full(len(studies), -1)
    rows[~bad] = np.arange(len(emb))
    pd.DataFrame({"study_id": studies, "row": rows, "unusable": bad}).to_parquet(
        DATA / "emb_next.parquet"
    )
    np.save(DATA / "emb_next.npy", emb)
    print(f"pool next ECGs embedded {emb.shape}, unusable {int(bad.sum())}")

    # Neighbours by the query ECGs' embeddings; the pool is the training queries themselves.
    cur = np.load(DATA / "emb.npy")
    row = pd.read_parquet(DATA / "emb.parquet").set_index("study_id").row
    q = pd.read_parquet(DATA / "queries.parquet")
    pool = train[(train.cur_study.map(row) >= 0).to_numpy()]
    pool = pool[pool.next_study.map(dict(zip(studies, rows, strict=True))).to_numpy() >= 0]
    p = torch.nn.functional.normalize(torch.as_tensor(cur[row[pool.cur_study].to_numpy()]), dim=1)
    for role in ("test", "val", "train"):
        r = q[(q.role == role) & (q.cur_study.map(row).fillna(-1) >= 0).to_numpy()]
        x = torch.nn.functional.normalize(torch.as_tensor(cur[row[r.cur_study].to_numpy()]), dim=1)
        sims = x @ p.T
        same = torch.as_tensor(r.subject_id.to_numpy()[:, None] == pool.subject_id.to_numpy()[None])
        sims[same] = -2.0
        top = sims.topk(K, dim=1)
        np.savez(
            DATA / f"neighbours_{role}.npz",
            query_study=r.cur_study.to_numpy(),
            pool_study=pool.cur_study.to_numpy()[top.indices.numpy()],
            sim=top.values.numpy(),
        )
        assert not (
            pool.subject_id.to_numpy()[top.indices.numpy()] == r.subject_id.to_numpy()[:, None]
        ).any()
        print(
            f"{role}: queries {len(r)}, pool {len(pool)}, mean top-{K} similarity "
            f"{float(top.values.mean()):.3f}"
        )


if __name__ == "__main__":
    {"extract": extract, "embed": embed}[sys.argv[1]]()
