import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tslm_memory.memory import (
    Memory,
    Part,
    Settings,
    Span,
    batch,
    grid_levels,
    pool_stats,
    pool_vectors,
)
from tslm_memory.tslm import add_lora

SETTINGS = Settings(levels=3, periods=(2, 2048, 16), calendar=True, exact=1)


def tiny_llm() -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    llm = LlamaForCausalLM(cfg).eval()
    llm.requires_grad_(False)
    return llm


def memory() -> Memory:
    return Memory(SETTINGS, entry_dim=48, n_stats=7, hidden=64, n_blocks=2, heads=2, head_dim=8)


def bank(sizes: tuple[int, int] = (5, 3), items: tuple[int, int] = (2, 1)):
    rng = np.random.default_rng(0)
    entries, notes = [], []
    for n, m in zip(sizes, items, strict=True):
        start = rng.integers(1, 150, n).astype(np.float32)
        entries.append(
            Part(
                torch.randn(n, 48),
                start,
                start - 1,
                np.zeros(n, bool),
                stats=rng.normal(size=(n, 7)).astype(np.float32),
                exact=rng.uniform(0, 100, (n, 1)).astype(np.float32),
                level=np.arange(n) % 3,
                date=15000 - start.astype(int),
            )
        )
        notes.append(Part(torch.randn(m, 64), start[:m], start[:m] - 1, np.zeros(m, bool)))
    return batch(entries, notes, [True, False], (48, 7, 1, 64), "cpu")


def test_zero_gates_leave_the_llm_unchanged():
    llm = tiny_llm()
    ids = torch.randint(0, 100, (2, 7))
    before = llm(ids).logits
    mem = memory()
    assert mem.attach(llm.model.layers) == [3, 7]
    mem.load(bank())
    assert torch.equal(llm(ids).logits, before)


def test_padded_entries_get_no_attention_and_gradients_reach_the_memory():
    llm = tiny_llm()
    mem = memory()
    mem.attach(llm.model.layers)
    for blk in mem.blocks:
        torch.nn.init.constant_(blk.attn_gate, 0.5)
    ids = torch.randint(0, 100, (2, 7))
    b = bank()
    assert b.slots[:, 0].tolist() == [True, False] and b.slots.shape == (2, 1 + 5 + 2)
    mem.load(b)
    out = llm(ids).logits
    # 1 null + 3 real entries + 1 item for sample 1
    attn = mem.blocks[0].last_attn[1].mean(0)
    assert torch.all(attn[:, 1 + 3 : 1 + 5] == 0) and torch.all(attn[:, -1] == 0)
    assert torch.all(attn[:, -2] > 0)
    out.sum().backward()
    b.vectors[1, 3:] = 100.0
    b.stats[1, 3:] = 100.0
    b.start[1, 3:] = 1
    mem.load(b)
    assert torch.allclose(llm(ids).logits, out)
    assert mem.blocks[0].q.weight.grad.abs().sum() > 0
    assert mem.proj[0].weight.grad.abs().sum() > 0
    assert all(mem.stats_proj[k][0].weight.grad.abs().sum() > 0 for k in range(3))
    assert all(mem.exact_in[k].weight.grad.abs().sum() > 0 for k in range(3))
    assert mem.span.mlp[0].weight.grad.abs().sum() > 0
    assert mem.text_proj[0].weight.grad.abs().sum() > 0


def test_lora_starts_at_zero_and_trains_beside_the_memory():
    llm = tiny_llm()
    ids = torch.randint(0, 100, (2, 7))
    before = llm(ids).logits
    lora = add_lora(llm, rank=4)
    mem = memory()
    mem.attach(llm.model.layers)
    mem.load(bank())
    out = llm(ids).logits
    assert torch.equal(out, before)
    assert all(p.dtype == torch.float32 and p.requires_grad for p in lora)
    out.sum().backward()
    b_grads = [p.grad for n, p in llm.named_parameters() if "lora_B" in n]
    assert len(b_grads) == 8 * 4
    assert all(g is not None and g.abs().sum() > 0 for g in b_grads)
    assert not any(p.requires_grad for n, p in llm.named_parameters() if "lora_" not in n)


def test_blocks_leave_positions_before_start_unchanged():
    llm = tiny_llm()
    ids = torch.randint(0, 100, (2, 7))
    before = llm(ids).logits
    mem = memory()
    mem.attach(llm.model.layers)
    for blk in mem.blocks:
        torch.nn.init.constant_(blk.attn_gate, 0.5)
        torch.nn.init.constant_(blk.ff_gate, 0.5)
    mem.load(bank())
    mem.start = torch.tensor([3, 5])
    out = llm(ids).logits
    assert torch.allclose(out[0, :3], before[0, :3]) and torch.allclose(out[1, :5], before[1, :5])
    assert not torch.allclose(out[0, 3:], before[0, 3:])
    assert not torch.allclose(out[1, 5:], before[1, 5:])


def test_grid_pyramid_pools_vectors_and_statistics_by_their_rules():
    g = grid_levels(37, 4, 3)
    assert g[g[:, 0] == 1].shape[0] == 9 and g[g[:, 0] == 2].tolist() == [[2, 0, 15], [2, 16, 31]]
    x = torch.arange(37.0)[:, None].repeat(1, 2)
    assert pool_vectors(x, g)[-1].tolist() == [23.5, 23.5]
    raw = np.stack(
        [np.arange(37.0), -np.arange(37.0), np.arange(37.0) % 5, 10 + np.arange(37.0)], 1
    )
    p = pool_stats(raw, ["sum", "min", "max", 2], g)
    assert p[0].tolist() == [6, -3, 3, 13]  # base 0 to 3, column 2 largest at base 3
    assert p[-1].tolist() == [sum(range(16, 32)), -31, 4, 29]  # column 2 first largest at base 19


def test_span_tells_one_second_apart_in_two_hours_and_one_day_from_a_week():
    torch.manual_seed(0)
    span = Span(64, Settings(levels=5, periods=(1, 8192, 14), calendar=False))
    t = torch.tensor([7000.0, 7001.0])
    e = span(t, t + 10, torch.zeros(2, dtype=torch.long))
    assert (e[0] - e[1]).abs().max() > 1e-4
    days = Span(64, SETTINGS)
    date = torch.tensor([15000, 15000])
    e = days(torch.tensor([30.0, 30.0]), torch.tensor([29.0, 23.0]), date)
    assert (e[0] - e[1]).abs().max() > 1e-4


def test_each_chunk_holds_the_slots_whose_midpoint_lies_in_it():
    from tslm_memory.memory import time_chunks

    # Null, then spans over 0 to 40 (midpoints 5, 15, 25, 35, 20) and one padded slot.
    lo = torch.tensor([[0.0, 0, 10, 20, 30, 0, 0]])
    hi = torch.tensor([[0.0, 10, 20, 30, 40, 40, 0]])
    mask = torch.tensor([[True] * 6 + [False]])
    allow = time_chunks(mask, lo, hi, 4)[0].int().tolist()
    assert allow == [
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0],
    ]


def test_mean_pool_reads_each_chunk_mean_after_the_null_entry():
    mem = Memory(SETTINGS, 48, 7, 64, 2, heads=2, head_dim=8, compress=4)
    b = bank((40, 40), (0, 0))
    with torch.no_grad():
        mem.load(b)
        full = Memory(SETTINGS, 48, 7, 64, 2, heads=2, head_dim=8)
        full.load_state_dict(mem.state_dict())
        full.load(b)
    assert mem.mem.shape == (2, 5, 64) and mem.mask[:, 0].all()
    assert torch.allclose(mem.mem[:, 0], full.mem[:, 0])
    # The pooled slots average to the mean of every real slot, weighted by chunk size.
    from tslm_memory.memory import time_chunks

    zero = torch.zeros(2, 1)
    lo = torch.cat([zero, torch.minimum(b.start, b.end), b.item_start], 1)
    hi = torch.cat([zero, torch.maximum(b.start, b.end), b.item_end], 1)
    inside = time_chunks(full.mask, lo, hi, 4)[0, :, 1:].float()
    want = (inside @ full.mem[0, 1:]) / inside.sum(-1, keepdim=True).clamp(min=1)
    assert torch.allclose(mem.mem[0, 1:][mem.mask[0, 1:]], want[inside.sum(-1) > 0], atol=1e-5)


def test_mean_pool_with_one_slot_per_chunk_reads_like_the_uncompressed_memory():
    llm = tiny_llm()
    torch.manual_seed(1)
    base = Memory(SETTINGS, 48, 7, 64, 2, heads=2, head_dim=8)
    pooled = Memory(SETTINGS, 48, 7, 64, 2, heads=2, head_dim=8, compress=4096)
    pooled.load_state_dict(base.state_dict())
    for m in (base, pooled):
        for blk in m.blocks:
            torch.nn.init.constant_(blk.attn_gate, 0.5)
            torch.nn.init.constant_(blk.ff_gate, 0.5)
    b = bank((5, 3), (0, 0))
    # Distinct spans, so that no two slots share a chunk.
    b.start[:] = torch.tensor([10.0, 30, 50, 70, 90])
    b.end[:] = b.start - 1
    ids = torch.randint(0, 100, (2, 7))
    outs = []
    for m in (base, pooled):
        hooks = [layer._forward_pre_hooks.copy() for layer in llm.model.layers]
        m.attach(llm.model.layers)
        m.load(b)
        outs.append(llm(ids).logits)
        for layer, h in zip(llm.model.layers, hooks, strict=True):
            layer._forward_pre_hooks.clear()
            layer._forward_pre_hooks.update(h)
    assert torch.allclose(outs[0], outs[1], atol=1e-4)


def test_mean_pool_evidence_is_the_chunk_holding_an_evidence_slot_or_the_null():
    mem = Memory(SETTINGS, 48, 7, 64, 2, heads=2, head_dim=8, compress=4)
    b = bank((5, 3), (0, 0))
    b.start[:] = torch.tensor([10.0, 30, 50, 70, 90])
    b.end[:] = b.start - 1
    b.slots[:] = False
    # Sample 0: its second entry, span 29 to 30, in chunk 1 of the range 9 to 90 (width 20.25).
    b.slots[0, 2] = True
    b.slots[1, 0] = True  # sample 1: no evidence slot, so the null
    with torch.no_grad():
        mem.load(b)
    ev = mem.evidence(b.slots)
    assert ev.shape == (2, 5)
    assert ev[0].tolist() == [False, False, True, False, False]
    assert ev[1].tolist() == [True, False, False, False, False]
    plain = Memory(SETTINGS, 48, 7, 64, 2, heads=2, head_dim=8)
    with torch.no_grad():
        plain.load(b)
    assert torch.equal(plain.evidence(b.slots), b.slots)
