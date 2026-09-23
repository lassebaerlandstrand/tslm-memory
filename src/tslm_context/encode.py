"""ECGFounder (MIT, NEJM AI 2025; github.com/PKUDigitalHealth/ECGFounder): one frozen 1024-d
embedding per 12-lead record, the only way a recording enters the reader."""

import numpy as np
import torch

from tslm_context.net1d import Net1D
from tslm_context.waves import DATA


def ecgfounder():
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
def embed(model, waves500: np.ndarray) -> np.ndarray:
    """Pooled features before ECGFounder's classifier. Input [N, 5000, 12] at 500 Hz in PTB-XL
    lead order, z-scored over the whole recording as in the authors' ptbxl_eval.py."""
    x = np.asarray(waves500, dtype=np.float32).transpose(0, 2, 1)
    x = (x - x.mean(axis=(1, 2), keepdims=True)) / (x.std(axis=(1, 2), keepdims=True) + 1e-8)
    _, h = model(torch.as_tensor(x, device="cuda"))
    return h.float().cpu().numpy()
