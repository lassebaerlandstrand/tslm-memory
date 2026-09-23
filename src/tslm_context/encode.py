"""Retrieval embeddings from the reader's own frozen series encoder (no extra model).

Each lead is z-scored as in the prompt, encoded, mean-pooled over patches, and the 12 lead vectors
are concatenated and L2-normalised, so cosine similarity compares whole 12-lead recordings.
"""

import numpy as np
import torch

from tslm_context.prompts import lead_series


@torch.no_grad()
def embed(model, waves: np.ndarray, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(waves), batch):
        z = np.stack([lead_series(np.asarray(w))[1] for w in waves[i : i + batch]])  # [B, 12, 1000]
        x = torch.as_tensor(z, device="cuda").reshape(-1, z.shape[-1])
        h = model.encoder(x).mean(dim=1).reshape(len(z), -1).float()
        out.append(torch.nn.functional.normalize(h, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def chronos2():
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")


@torch.no_grad()
def embed_chronos2(pipe, waves: np.ndarray, batch: int = 32) -> np.ndarray:
    """Chronos-2 encoder: 12 leads as one multivariate series, patch embeddings mean-pooled per
    lead, the lead vectors concatenated and L2-normalised."""
    out = []
    for i in range(0, len(waves), batch):
        x = np.stack([lead_series(np.asarray(w))[1] for w in waves[i : i + batch]])  # [B, 12, 1000]
        embs, _ = pipe.embed(x, batch_size=batch * 12)
        h = torch.stack([e[:, 1:-1].float().mean(dim=1).reshape(-1) for e in embs])
        out.append(torch.nn.functional.normalize(h, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def ecgfounder():
    from tslm_context.net1d import Net1D
    from tslm_context.waves import DATA

    model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=150,
    )
    ckpt = torch.load(
        DATA / "ecgfounder" / "12_lead_ECGFounder.pth", map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt["state_dict"])
    model.return_features = True
    return model.cuda().eval()


@torch.no_grad()
def embed_ecgfounder(model, waves500: np.ndarray, batch: int = 64) -> np.ndarray:
    """ECGFounder's pooled features before its classifier. Input [N, 5000, 12] at 500 Hz in PTB-XL
    lead order, z-scored over the whole recording as in the authors' ptbxl_eval.py."""
    out = []
    for i in range(0, len(waves500), batch):
        x = np.asarray(waves500[i : i + batch], dtype=np.float32).transpose(0, 2, 1)
        x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
        _, h = model(torch.as_tensor(x, device="cuda"))
        out.append(torch.nn.functional.normalize(h.float(), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)
