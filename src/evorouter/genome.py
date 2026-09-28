"""Genome <-> per-layer router biases.

A genome theta is a flat vector of length ``len(layers) * num_experts``, laid out layer-major:
theta[j * E + i] is the bias of expert i in the j-th adapted layer. Optional per-layer ``scale``
maps search units to logit units (we search in units of the median top-k margin Delta_l, so
beta_{l,i} = Delta_l * theta_{l,i}).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class BiasGenome:
    layers: tuple[int, ...]
    num_experts: int
    scale: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("need at least one adapted layer")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"duplicate layers in {self.layers}")
        if self.scale is not None and len(self.scale) != len(self.layers):
            raise ValueError(f"scale has {len(self.scale)} entries for {len(self.layers)} layers")

    @property
    def dim(self) -> int:
        return len(self.layers) * self.num_experts

    def zeros(self) -> np.ndarray:
        return np.zeros(self.dim)

    def biases(
        self, theta: np.ndarray | Tensor | Sequence[float], device: str | torch.device = "cpu"
    ) -> dict[int, Tensor]:
        """theta of shape [dim] -> {layer: [E]}; shape [N, dim] -> {layer: [N, E]} (float32)."""
        t = torch.as_tensor(np.asarray(theta, dtype=np.float32), device=device)
        if t.shape[-1] != self.dim or t.dim() not in (1, 2):
            raise ValueError(f"theta shape {tuple(t.shape)} incompatible with genome dim {self.dim}")
        t = t.reshape(*t.shape[:-1], len(self.layers), self.num_experts)
        if self.scale is not None:
            t = t * torch.tensor(self.scale, dtype=t.dtype, device=t.device)[:, None]
        return {layer: t[..., j, :] for j, layer in enumerate(self.layers)}
