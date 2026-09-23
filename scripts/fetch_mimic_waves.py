"""Download the selected MIMIC-IV-ECG records (.hea and .dat) with a few threads.

Files already on disk are skipped, so the script resumes after an interruption.
"""

import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from tslm_context.waves import DATA

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0/"


def fetch(rel: str) -> int:
    out = DATA / "mimic" / rel
    if out.exists():
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(30):
        try:
            with urllib.request.urlopen(BASE + rel, timeout=60) as r:
                data = r.read()
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(out)
            return len(data)
        except OSError:
            time.sleep(min(5 * (attempt + 1), 60))
    raise RuntimeError(f"failed {rel}")


def main(threads: int = 12) -> None:
    sel = pd.read_parquet(DATA / "mimic_selection.parquet")
    order = {"test": 0, "train": 1, "index": 2}
    paths = sel.assign(o=sel.role.map(order)).sort_values("o").path
    files = [f"{p}.{ext}" for p in paths for ext in ("hea", "dat")]
    t, done, total = time.time(), 0, 0
    with ThreadPoolExecutor(threads) as ex:
        for n in ex.map(fetch, files):
            done += 1
            total += n
            if done % 500 == 0:
                rate = total / 1024 / (time.time() - t)
                print(
                    f"{done}/{len(files)} files {total / 2**20:.0f} MB {rate:.0f} KB/s", flush=True
                )
    print("done", len(files))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
