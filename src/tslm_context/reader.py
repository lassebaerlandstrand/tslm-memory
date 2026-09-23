"""The reader: Llama-3.2-1B with OpenTSLM's released ECG SoftPrompt LoRA, reading recordings as
16 tokens each.

OpenTSLM's own ECG encoder gives the same output for every record (spread across records about
3e-7 against a mean of 0.07; after the bf16 cast 48,000 soft tokens take 1,647 distinct values),
so a recording enters instead as its frozen ECGFounder embedding, a small trained MLP to 16 x 128,
and OpenTSLM's frozen projector to the LLM's embedding space. One BOS per sample.
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch import nn

REPO = "OpenTSLM/llama-3.2-1b-ecg-sp"
N_TOKENS = 16


def load(checkpoint: str | None = None):
    load_dotenv()
    from opentslm.model.llm.OpenTSLM import OpenTSLM

    model = OpenTSLM.load_pretrained(REPO, device="cuda", enable_lora=True)
    if checkpoint is not None:
        model.load_from_file(checkpoint)
    # OpenTSLM hard-codes eager attention; SDPA computes the same function with far less memory.
    model.llm.set_attn_implementation("sdpa")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


class Adapter(nn.Module):
    """1024-d ECGFounder embedding -> N_TOKENS x 128, the input size of OpenTSLM's projector."""

    def __init__(self, d_in: int = 1024, hidden: int = 1024, d_out: int = 128):
        super().__init__()
        self.d_out = d_out
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_TOKENS * d_out),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e).view(-1, N_TOKENS, self.d_out)


def _text(model, text: str, bos: bool) -> torch.Tensor:
    ids = model.tokenizer(text, add_special_tokens=bos, return_tensors="pt").input_ids[0]
    return model.llm.get_input_embeddings()(ids.to(model.device))


def embed(model, adapter: Adapter, table: torch.Tensor, s: dict) -> torch.Tensor:
    """One sample's segments -> [L, H]. table: [N, 1024] ECGFounder embeddings on the GPU."""
    rows = [x for kind, x in s["segments"] if kind == "ecg"]
    dtype = model.llm.get_input_embeddings().weight.dtype
    rec = model.projector(adapter(table[rows])).to(dtype)  # [n, N_TOKENS, H]
    parts, j = [], 0
    for kind, x in s["segments"]:
        if kind == "text":
            parts.append(_text(model, x, bos=not parts))
        else:
            parts.append(rec[j])
            j += 1
    return torch.cat(parts)


def embed_left(model, adapter: Adapter, table: torch.Tensor, batch: list[dict]):
    embs = [embed(model, adapter, table, s) for s in batch]
    n = max(e.size(0) for e in embs)
    out = embs[0].new_zeros(len(embs), n, embs[0].size(1))
    att = torch.zeros(len(embs), n, dtype=torch.long, device=out.device)
    for i, e in enumerate(embs):
        out[i, n - e.size(0) :] = e
        att[i, n - e.size(0) :] = 1
    return out, att


def last_logits(model, adapter: Adapter, table: torch.Tensor, batch: list[dict]) -> torch.Tensor:
    """Next-token logits after each sample's last segment ("Answer:"), [B, vocab]."""
    emb, att = embed_left(model, adapter, table, batch)
    # Left padding needs positions that start at the first real token.
    pos = (att.cumsum(dim=1) - 1).clamp(min=0)
    out = model.llm(inputs_embeds=emb, attention_mask=att, position_ids=pos, logits_to_keep=1)
    return out.logits[:, -1].float()


def yes_no(model) -> tuple[int, int]:
    tok = model.tokenizer
    return (
        tok(" yes", add_special_tokens=False).input_ids[0],
        tok(" no", add_special_tokens=False).input_ids[0],
    )


@torch.no_grad()
def p_yes(model, adapter: Adapter, table: torch.Tensor, batch: list[dict]) -> list[float]:
    """P(yes) / (P(yes) + P(no)) for the token after "Answer:"."""
    logits = last_logits(model, adapter, table, batch)
    yes, no = yes_no(model)
    return logits[:, [yes, no]].softmax(dim=1)[:, 0].tolist()


def loss(model, adapter: Adapter, table: torch.Tensor, batch: list[dict]) -> torch.Tensor:
    """Cross-entropy of the answer token at the position p_yes scores."""
    yes, no = yes_no(model)
    target = torch.tensor([yes if s["answer"] == " yes" else no for s in batch], device="cuda")
    return F.cross_entropy(last_logits(model, adapter, table, batch), target)


def lm_loss(model, adapter: Adapter, table: torch.Tensor, batch: list[dict]) -> torch.Tensor:
    """Next-token loss over a multi-token target (stage 1: the recording's own machine read),
    appended without BOS and closed with EOS. Left padding puts every target at the end, so only
    the last positions' logits are computed (all positions would need about 3 GB at batch 64)."""
    tok = model.tokenizer
    targets = [
        tok(s["target"], add_special_tokens=False).input_ids + [tok.eos_token_id] for s in batch
    ]
    embs = []
    for s, ids in zip(batch, targets, strict=True):
        ids = torch.tensor(ids, device="cuda")
        embs.append(
            torch.cat([embed(model, adapter, table, s), model.llm.get_input_embeddings()(ids)])
        )
    n = max(e.size(0) for e in embs)
    x = embs[0].new_zeros(len(embs), n, embs[0].size(1))
    att = torch.zeros(len(embs), n, dtype=torch.long, device=x.device)
    for i, e in enumerate(embs):
        x[i, n - e.size(0) :], att[i, n - e.size(0) :] = e, 1
    keep = max(len(t) for t in targets) + 1
    # The logit at window position j predicts the token at j + 1.
    y = torch.full((len(embs), keep), -100, device=x.device)
    for i, t in enumerate(targets):
        y[i, keep - 1 - len(t) : keep - 1] = torch.tensor(t, device=x.device)
    pos = (att.cumsum(dim=1) - 1).clamp(min=0)
    logits = model.llm(
        inputs_embeds=x, attention_mask=att, position_ids=pos, logits_to_keep=keep
    ).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def generate(
    model, adapter: Adapter, table: torch.Tensor, batch: list[dict], max_new_tokens: int = 60
) -> list[str]:
    emb, att = embed_left(model, adapter, table, batch)
    ids = model.llm.generate(
        inputs_embeds=emb,
        attention_mask=att,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    return model.tokenizer.batch_decode(ids, skip_special_tokens=True)
