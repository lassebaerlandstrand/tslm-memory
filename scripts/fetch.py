"""Fetch the records in data/fetch.txt from a Kaggle copy of MIMIC-IV-ECG 1.0.

Per-file requests to Kaggle are rate-limited, so the whole copy (a 34 GB zip, resumable) is
downloaded once and only the listed records are extracted. Every extracted file is checked against
PhysioNet's official SHA256SUMS.txt (data/mimic/SHA256SUMS.txt); if that is not on disk yet, the
copy's own list is used and a rerun cross-checks against the official one. Needs KAGGLE_API_TOKEN
in .env. Usage: uv run --group fetch python scripts/fetch.py
"""

import hashlib
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

DATA = Path(__file__).resolve().parents[1] / "data"
DATASET = "subhashhenry/mimic-iv-ecg-demo-diagnostic-ecg-matched-subset"
ZIP = DATA / "kaggle" / "mimic-iv-ecg-demo-diagnostic-ecg-matched-subset.zip"
PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"


def read_sums(text: str) -> dict[str, str]:
    return dict(reversed(line.split(maxsplit=1)) for line in text.splitlines())


def main() -> None:
    load_dotenv(DATA.parent / ".env")
    if not ZIP.exists() or ZIP.with_name(ZIP.name + ".kaggle-partial").exists():
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(DATASET, path=str(ZIP.parent), quiet=False)
    files = (DATA / "fetch.txt").read_text().split()
    official = DATA / "mimic" / "SHA256SUMS.txt"
    with zipfile.ZipFile(ZIP) as z:
        mirror = read_sums(z.read(PREFIX + "SHA256SUMS.txt").decode())
        sums = read_sums(official.read_text()) if official.exists() else mirror
        todo = [f for f in files if not (DATA / "mimic" / f).exists()]
        print(f"files {len(files)}, to extract {len(todo)}", file=sys.stderr, flush=True)
        for i, f in enumerate(todo, 1):
            data = z.read(PREFIX + f)
            assert hashlib.sha256(data).hexdigest() == sums[f], "checksum mismatch"
            out = DATA / "mimic" / f
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_suffix(out.suffix + ".tmp").write_bytes(data)
            out.with_suffix(out.suffix + ".tmp").rename(out)
            if i % 5000 == 0:
                print(f"{i}/{len(todo)} extracted", file=sys.stderr, flush=True)
    if not official.exists():
        print("done; rerun once data/mimic/SHA256SUMS.txt is on disk", file=sys.stderr)
        return
    ref = read_sums(official.read_text())
    bad = sum(ref.get(f) != mirror.get(f) for f in files)
    # Files fetched earlier from PhysioNet itself are checked too.
    wrong = sum(
        hashlib.sha256((DATA / "mimic" / f).read_bytes()).hexdigest() != ref[f] for f in files
    )
    print(
        f"done; {len(files)} files; {bad} mirror checksums and {wrong} files on disk differ "
        "from PhysioNet's",
        file=sys.stderr,
    )
    assert bad == 0 and wrong == 0


if __name__ == "__main__":
    main()
