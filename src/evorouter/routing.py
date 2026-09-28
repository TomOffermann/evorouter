"""Routing interventions on OLMoE-style sparse MoE blocks.

This is the only module that touches HuggingFace internals. It swaps the forward of selected
``OlmoeSparseMoeBlock`` instances for :func:`routed_moe_forward`, a line-by-line copy of the HF
implementation (transformers 4.57) with one change: a per-expert bias ``b`` enters the top-k.

Notation (DeepSeekMoE terms): router logits z = u^T e_i, probabilities p = softmax(z).

* ``mode="selection"`` (genome G1): selected set S = TopK(softmax(z + b)); gates = p_i for i in S.
  The bias decides *which* experts run, never *how much* they are weighted.
* ``mode="logit"`` (control G1'): S = TopK(softmax(z + b)); gates = softmax(z + b)_i.
  The bias moves the gates too, so it is gradient-visible.

With ``b = 0`` both modes perform exactly the HF computation (tested bit-for-bit).
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

Mode = Literal["selection", "logit"]


@dataclass
class RouterRecord:
    """What one routed MoE block did in one forward pass (flattened over batch x sequence)."""

    logits: Tensor  # [tokens, E] raw router logits z, float32
    selected: Tensor  # [tokens, k] expert indices
    weights: Tensor  # [tokens, k] gates actually applied, float32


@dataclass
class RoutingIntervention:
    """Per-layer selection biases for one forward pass.

    ``biases[layer]`` has shape ``[E]`` (same bias for every row) or ``[B, E]`` (one bias per batch
    row, which is how a population of genomes is evaluated in one batch). ``position_mask`` of
    shape ``[B, T]`` restricts the bias to marked tokens; unmarked tokens are routed as in the base
    model. Layer indices are 0-based.
    """

    biases: dict[int, Tensor]
    mode: Mode = "selection"
    position_mask: Tensor | None = None
    record: bool = False
    records: dict[int, RouterRecord] = field(default_factory=dict)


def moe_blocks(model: nn.Module) -> list[nn.Module]:
    """The sparse MoE block of every decoder layer, validated against our assumptions."""
    if getattr(model.config, "model_type", None) != "olmoe":
        raise NotImplementedError(f"routing is implemented for OLMoE only, got {model.config.model_type!r}")
    blocks = [layer.mlp for layer in model.model.layers]
    for i, block in enumerate(blocks):
        missing = [
            a for a in ("gate", "experts", "top_k", "num_experts", "norm_topk_prob") if not hasattr(block, a)
        ]
        if missing:
            raise TypeError(f"layer {i}: MoE block {type(block).__name__} lacks {missing}")
    return blocks


def last_layers(model: nn.Module, n: int) -> tuple[int, ...]:
    """0-based indices of the last ``n`` MoE layers."""
    num = len(moe_blocks(model))
    if not 1 <= n <= num:
        raise ValueError(f"n must be in [1, {num}], got {n}")
    return tuple(range(num - n, num))


def _expand_bias(bias: Tensor, batch: int, seq: int, mask: Tensor | None) -> Tensor:
    """Bias of shape [E] or [B, E] -> [B*T, E], zeroed where ``mask`` is False."""
    if bias.dim() == 1:
        b = bias.expand(batch * seq, -1)
    elif bias.dim() == 2 and bias.shape[0] == batch:
        b = bias.repeat_interleave(seq, dim=0)
    else:
        raise ValueError(f"bias shape {tuple(bias.shape)} does not match batch size {batch}")
    if mask is not None:
        if mask.shape != (batch, seq):
            raise ValueError(f"position_mask shape {tuple(mask.shape)} != {(batch, seq)}")
        b = b * mask.reshape(-1, 1).to(b.dtype)
    return b


def routed_moe_forward(
    block: nn.Module,
    hidden_states: Tensor,
    bias: Tensor,
    mode: Mode,
    position_mask: Tensor | None,
    record: dict[int, RouterRecord] | None = None,
    layer: int | None = None,
) -> tuple[Tensor, Tensor]:
    """OlmoeSparseMoeBlock.forward with a bias on the expert selection (see module docstring)."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = block.gate(hidden_states)

    probs = F.softmax(router_logits, dim=1, dtype=torch.float)
    b = _expand_bias(
        bias.to(device=probs.device, dtype=torch.float), batch_size, sequence_length, position_mask
    )
    biased_probs = F.softmax(router_logits.float() + b, dim=1)
    _, selected_experts = torch.topk(biased_probs, block.top_k, dim=-1)

    source = probs if mode == "selection" else biased_probs
    routing_weights = source.gather(1, selected_experts)
    if block.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    if record is not None and layer is not None:
        record[layer] = RouterRecord(
            router_logits.detach().float(), selected_experts, routing_weights.detach()
        )
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )
    expert_mask = F.one_hot(selected_experts, num_classes=block.num_experts).permute(2, 1, 0)
    for expert_idx in range(block.num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = block.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


@contextmanager
def apply_routing(model: nn.Module, intervention: RoutingIntervention) -> Iterator[RoutingIntervention]:
    """Route the layers in ``intervention.biases`` through :func:`routed_moe_forward` inside the block."""
    if intervention.mode not in ("selection", "logit"):
        raise ValueError(f"unknown mode {intervention.mode!r}")
    blocks = moe_blocks(model)
    bad = [layer for layer in intervention.biases if not 0 <= layer < len(blocks)]
    if bad:
        raise ValueError(f"layer indices {bad} out of range [0, {len(blocks)})")
    intervention.records.clear()
    record = intervention.records if intervention.record else None
    patched = []
    try:
        for layer, bias in intervention.biases.items():
            block = blocks[layer]

            def forward(self, hidden_states, _bias=bias, _layer=layer):
                return routed_moe_forward(
                    self, hidden_states, _bias, intervention.mode, intervention.position_mask, record, _layer
                )

            block.forward = types.MethodType(forward, block)
            patched.append(block)
        yield intervention
    finally:
        for block in patched:
            del block.forward  # falls back to the class-level HF forward
