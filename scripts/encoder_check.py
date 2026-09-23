"""Does OpenTSLM's released ECG SoftPrompt encoder respond to the ECG?

Encodes 64 test-query ECGs (12 leads, z-scored, 100 Hz as OpenTSLM reads them) with the released
encoder and projector and reports the spread of the outputs across all tokens against their
magnitude, and how many distinct soft tokens remain after the cast to the LLM's bf16. Writes
runs/encoder_check.json. CPU only.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model.projector.MLPProjector import MLPProjector

from tslm_context.reader import REPO
from tslm_context.waves import DATA, read500


def main(n: int = 64) -> None:
    ckpt = torch.load(
        glob.glob(f"{snapshot_download(REPO)}/*.pt")[0], map_location="cpu", weights_only=False
    )
    enc = TransformerCNNEncoder().eval()
    enc.load_state_dict(ckpt["encoder_state"])
    proj = MLPProjector(128, 2048, "cpu").eval()
    proj.load_state_dict(ckpt["projector_state"])
    q = pd.read_parquet(DATA / "queries.parquet")
    studies = q[q.role == "test"].cur_study.to_numpy()[:n]
    ecgs = pd.read_parquet(DATA / "ecgs.parquet", columns=["study_id", "path"])
    path = dict(zip(ecgs.study_id, ecgs.path, strict=True))
    x = []
    for s in studies:
        w = np.nan_to_num(read500(path[s]))[::5][:1000]
        x.append(((w - w.mean(0)) / np.where(w.std(0) > 1e-6, w.std(0), 1.0)).T)
    x = torch.as_tensor(np.stack(x), dtype=torch.float32).reshape(-1, 1000)
    with torch.no_grad():
        h = enc(x)  # [n * 12, 250, 128]
        p = proj(h).reshape(-1, 2048)
        noise = proj(enc(torch.randn(48, 1000))).reshape(-1, 2048)
    tokens = p.to(torch.bfloat16)
    res = {
        "n_ecgs": len(studies),
        "n_soft_tokens": int(tokens.shape[0]),
        "encoder output magnitude": float(h.abs().mean()),
        "encoder output spread across tokens": float(h.reshape(-1, 128).std(0).median()),
        "distinct bf16 soft tokens": int(torch.unique(tokens.float(), dim=0).shape[0]),
        "final layernorm gain": float(
            ckpt["encoder_state"]["encoder.layers.5.norm2.weight"].abs().mean()
        ),
        "noise vs ecg soft-token distance ratio": float(
            (noise.mean(0) - p.mean(0)).norm() / p.mean(0).norm()
        ),
    }
    Path("runs/encoder_check.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
