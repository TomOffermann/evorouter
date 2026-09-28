"""Shared experiment setup for scripts: model, data, search/validation split, adapted layers, margins."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from evorouter.diagnostics import router_logits_at_scored_positions, topk_margins
from evorouter.models import OLMOE_ID, byte_encode, hf_encoder, load_model, tiny_olmoe
from evorouter.routing import last_layers
from evorouter.scoring import Encode, build_requests
from evorouter.tasks.base import MCQuestion, search_val_split


def tiny_data() -> dict[str, list[MCQuestion]]:
    """Synthetic questions for offline --tiny runs."""
    base = [
        ("Which is renewable?", ("coal", "wind", "oil", "gas"), 1),
        ("What do plants need?", ("light", "sand", "rock"), 0),
        ("Largest planet?", ("Mars", "Venus", "Jupiter", "Earth"), 2),
    ]
    pool = [MCQuestion(f"t{i}", q + f" ({i})", c, a) for i, (q, c, a) in enumerate(base * 4)]
    return {"train": pool, "validation": pool[:3], "test": pool[:6]}


@dataclass
class Context:
    model: nn.Module
    encode: Encode
    data: dict[str, list[MCQuestion]]
    search: list[MCQuestion]
    val: list[MCQuestion]
    layers: tuple[int, ...]
    margins: tuple[float, ...]  # median top-k margin per adapted layer on the search set

    @property
    def num_experts(self) -> int:
        return self.model.config.num_experts

    @property
    def top_k(self) -> int:
        return self.model.config.num_experts_per_tok


def load_context(
    *,
    seed: int = 0,
    n_search: int = 64,
    n_val: int = 256,
    adapted_layers: int = 4,
    tiny: bool = False,
    model_id: str = OLMOE_ID,
) -> Context:
    if tiny:
        model, encode, data = tiny_olmoe(), byte_encode, tiny_data()
        n_search, n_val, adapted_layers = min(n_search, 4), min(n_val, 4), min(adapted_layers, 2)
    else:
        from evorouter.tasks.arc import load_arc

        model, tokenizer = load_model(model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        encode, data = hf_encoder(tokenizer), load_arc("ARC-Challenge")
    search, val = search_val_split(data["train"], seed, n_search, n_val)
    layers = last_layers(model, adapted_layers)
    logits = router_logits_at_scored_positions(model, build_requests(search, encode), layers)
    stats = topk_margins(logits, model.config.num_experts_per_tok)
    margins = tuple(stats[layer]["median"] for layer in layers)
    return Context(model, encode, data, search, val, layers, margins)


def shard(items: list, index: int, count: int) -> list:
    """Round-robin share ``index`` of ``count`` (one share per GPU)."""
    if not 0 <= index < count:
        raise ValueError(f"shard index {index} not in [0, {count})")
    return items[index::count]
