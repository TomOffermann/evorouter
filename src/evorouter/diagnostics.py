"""Router statistics of the base model: top-k margins (sets the CMA-ES step size) and loads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from evorouter.evaluate import pack_rows
from evorouter.routing import moe_blocks
from evorouter.scoring import Request


@torch.no_grad()
def router_logits_at_scored_positions(
    model: nn.Module, requests: Sequence[Request], layers: Sequence[int], max_batch_tokens: int = 16384
) -> dict[int, torch.Tensor]:
    """Base-model router logits z at every scored position: {layer: [n_positions, E]} (float32, CPU)."""
    blocks = moe_blocks(model)
    device = next(model.parameters()).device
    captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    current: dict[int, torch.Tensor] = {}
    handles = [
        blocks[layer].gate.register_forward_hook(lambda _m, _i, out, _l=layer: current.__setitem__(_l, out))
        for layer in layers
    ]
    try:
        for batch in pack_rows([len(r.tokens) for r in requests], max_batch_tokens):
            reqs = [requests[i] for i in batch]
            max_len = max(len(r.tokens) for r in reqs)
            ids = torch.zeros((len(reqs), max_len), dtype=torch.long)
            attn = torch.zeros_like(ids)
            flat_idx = []
            for bi, r in enumerate(reqs):
                ids[bi, : len(r.tokens)] = torch.tensor(r.tokens)
                attn[bi, : len(r.tokens)] = 1
                flat_idx += [bi * max_len + p for p in r.scored_positions]
            model(input_ids=ids.to(device), attention_mask=attn.to(device), use_cache=False)
            sel = torch.tensor(flat_idx, device=device)
            for layer in layers:
                captured[layer].append(
                    current[layer].reshape(-1, current[layer].shape[-1])[sel].float().cpu()
                )
    finally:
        for h in handles:
            h.remove()
    return {layer: torch.cat(chunks) for layer, chunks in captured.items()}


def topk_margins(logits: dict[int, torch.Tensor], k: int) -> dict[int, dict[str, float]]:
    """Per layer: quantiles of z_(k) - z_(k+1), the logit gap a bias must overcome to flip top-k."""
    out = {}
    for layer, z in logits.items():
        top = z.topk(k + 1, dim=-1).values
        gap = (top[:, k - 1] - top[:, k]).numpy()
        out[layer] = {
            "median": float(np.median(gap)),
            "q10": float(np.quantile(gap, 0.1)),
            "q90": float(np.quantile(gap, 0.9)),
            "logit_std": float(z.std()),
            "n_positions": int(z.shape[0]),
        }
    return out


def expert_loads(logits: dict[int, torch.Tensor], k: int) -> dict[int, dict[str, float]]:
    """Per layer: share of top-k slots each expert receives, and MaxVio = (max load - mean) / mean."""
    out = {}
    for layer, z in logits.items():
        counts = torch.bincount(z.topk(k, dim=-1).indices.flatten(), minlength=z.shape[-1]).double()
        mean = counts.mean()
        out[layer] = {
            "max_vio": float((counts.max() - mean) / mean),
            "unused_experts": int((counts == 0).sum()),
        }
    return out
