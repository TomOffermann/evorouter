"""Router statistics: top-k margins (CMA-ES step size), expert usage and loads, and how many
routing decisions a genome actually changes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from evorouter.evaluate import pack_rows
from evorouter.routing import Mode, RoutingIntervention, apply_routing, moe_blocks
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


def expert_usage(logits: dict[int, torch.Tensor], k: int) -> dict[int, np.ndarray]:
    """Per layer: fraction of positions at which each expert is in the base top-k, shape [E]."""
    out = {}
    for layer, z in logits.items():
        counts = torch.bincount(z.topk(k, dim=-1).indices.flatten(), minlength=z.shape[-1]).double()
        out[layer] = (counts / z.shape[0]).numpy()
    return out


@torch.no_grad()
def selection_change_rate(
    model: nn.Module,
    requests: Sequence[Request],
    biases: dict[int, torch.Tensor],
    mode: Mode = "selection",
    scope: str = "scored",
    max_batch_tokens: int = 16384,
) -> dict[int, float]:
    """Per layer: fraction of in-scope positions whose top-k *set* differs from the base model.

    ``biases`` are single-genome biases {layer: [E]}. With scope="all", changes in earlier adapted
    layers propagate, so later layers also count selections that changed only indirectly.
    """
    device = next(model.parameters()).device
    changed = {layer: 0 for layer in biases}
    total = 0
    zeros = {layer: torch.zeros_like(b) for layer, b in biases.items()}
    for batch in pack_rows([len(r.tokens) for r in requests], max_batch_tokens):
        reqs = [requests[i] for i in batch]
        max_len = max(len(r.tokens) for r in reqs)
        ids = torch.zeros((len(reqs), max_len), dtype=torch.long)
        attn = torch.zeros_like(ids)
        mask = torch.zeros((len(reqs), max_len), dtype=torch.bool)
        for bi, r in enumerate(reqs):
            ids[bi, : len(r.tokens)] = torch.tensor(r.tokens)
            attn[bi, : len(r.tokens)] = 1
            if scope == "scored":
                mask[bi, list(r.scored_positions)] = True
            else:
                mask[bi, : len(r.tokens)] = True
        ids, attn, mask = ids.to(device), attn.to(device), mask.to(device)
        selections = []
        for b in (zeros, biases):
            iv = RoutingIntervention(b, mode, position_mask=mask, record=True)
            with apply_routing(model, iv):
                model(input_ids=ids, attention_mask=attn, use_cache=False)
            selections.append({layer: rec.selected.sort(-1).values for layer, rec in iv.records.items()})
        flat = mask.reshape(-1)
        total += int(flat.sum())
        for layer in biases:
            diff = (selections[0][layer] != selections[1][layer]).any(-1)
            changed[layer] += int((diff & flat).sum())
    return {layer: changed[layer] / max(total, 1) for layer in biases}
