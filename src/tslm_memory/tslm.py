"""The host TSLM, OpenTSLM-SoftPrompt on Llama-3.2-1B from the TSQA stage, and its answer loss."""

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from peft import LoraConfig, inject_adapter_in_model
from torch.nn.utils.rnn import pad_sequence

from tslm_memory.memory import Memory

HOST = "OpenTSLM/llama-3.2-1b-tsqa-sp"


def load_host() -> OpenTSLMSP:
    host = OpenTSLMSP(llm_id="meta-llama/Llama-3.2-1B", device="cuda")
    # OpenTSLM loads the LLM with eager attention, which keeps every layer's full score matrix.
    # SDPA computes the same attention without it.
    host.llm.set_attn_implementation("sdpa")
    host.load_from_file(hf_hub_download(HOST, "model_checkpoint.pt"))
    host.eval().requires_grad_(False)
    return host


def add_lora(llm: torch.nn.Module, rank: int) -> list[torch.nn.Parameter]:
    """LoRA on q, k, v and o in every layer, in place. B starts at zero, so the LLM is unchanged
    until it trains. Adapter weights are kept in float32 while the LLM stays in bf16."""
    cfg = LoraConfig(
        r=rank, lora_alpha=2 * rank, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    inject_adapter_in_model(cfg, llm)
    params = [p for n, p in llm.named_parameters() if "lora_" in n]
    for p in params:
        p.data = p.data.float()
    return params


def prompts(questions: list[str], days: list[tuple[str, torch.Tensor]]) -> list[dict]:
    """The TSLM's own input: today's 48 readings formatted as in OpenTSLM TSQA, then the question.

    days holds per sample today's (opening sentence, kWh [48]). The question goes last, so the
    memory is queried from the answer position. Today's total is given as text, as the memory
    entries give it for past days. Missing readings count as 0."""
    out = []
    for q, (opening, x) in zip(questions, days, strict=True):
        x = torch.nan_to_num(x)
        m, s = x.mean(), x.std()
        text = (
            f"{opening}. It has total {48 * m.item():.1f} kWh, mean {m.item():.4f} and "
            f"std {s.item():.4f}."
        )
        out.append(
            {
                "pre_prompt": "You are given a household's electricity use.",
                "time_series_text": [text],
                "time_series": [(x - m) / (s + 1e-8)],
                "post_prompt": f"{q} Answer:",
            }
        )
    return out


def answer_logits(
    host: OpenTSLMSP, batch: list[dict], answers: list[str], memory: Memory
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logits and targets at the answer positions only. The memory blocks act from the question
    onward.

    Returns logits [M, V], targets [M], the sample of each position [M], and per sample the
    position that predicts the first answer token [B].

    OpenTSLM's compute_loss labels BOS and padding and appends answers after the batch padding,
    so the answer part is built here on top of its own series input path."""
    embeds, mask = host.pad_and_apply_batch(batch)
    tok, embed = host.tokenizer, host.llm.get_input_embeddings()
    lengths = mask.sum(1)
    # The post prompt is the question, tokenized alone as OpenTSLM does, at each prompt's end.
    post = [len(tok(b["post_prompt"]).input_ids) for b in batch]
    memory.start = lengths - torch.tensor(post, device=lengths.device)
    seqs, labels = [], []
    for b, ans in enumerate(answers):
        # TSQA training put BOS before every answer, so BOS goes in as input and is not a target.
        ids = [tok.bos_token_id, *tok(ans, add_special_tokens=False).input_ids, tok.eos_token_id]
        ids = torch.tensor(ids, device=embeds.device)
        seq = torch.cat([embeds[b, : lengths[b]], embed(ids)])
        lab = torch.full((len(seq),), -100, device=embeds.device)
        lab[lengths[b] : -1] = ids[1:]
        seqs.append(seq)
        labels.append(lab)
    inputs = pad_sequence(seqs, batch_first=True)
    attn = pad_sequence([torch.ones(len(s), dtype=torch.long) for s in seqs], batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    hidden = host.llm.model(inputs_embeds=inputs, attention_mask=attn.to(inputs.device))
    sel = labels != -100
    logits = host.llm.lm_head(hidden.last_hidden_state[sel]).float()
    return logits, labels[sel], sel.nonzero()[:, 0], lengths


def loss_and_exact(
    logits: torch.Tensor, targets: torch.Tensor, sample: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Answer-token loss and per-sample exact match.

    Every answer token being the argmax under teacher forcing is the same as greedy decoding
    producing exactly the answer and then EOS."""
    wrong = torch.zeros(n, device=logits.device)
    wrong.index_add_(0, sample, (logits.argmax(-1) != targets).float())
    return F.cross_entropy(logits, targets), wrong == 0
