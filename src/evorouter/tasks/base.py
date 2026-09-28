"""Task-independent multiple-choice question type and data splits."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MCQuestion:
    id: str
    question: str
    choices: tuple[str, ...]
    answer: int  # index into choices

    def __post_init__(self) -> None:
        if not 0 <= self.answer < len(self.choices):
            raise ValueError(f"{self.id}: answer {self.answer} out of range for {len(self.choices)} choices")


def search_val_split(
    pool: Sequence[MCQuestion], seed: int, n_search: int, n_val: int
) -> tuple[list[MCQuestion], list[MCQuestion]]:
    """Disjoint random search and validation sets drawn from ``pool`` (e.g. a train split)."""
    if n_search + n_val > len(pool):
        raise ValueError(f"requested {n_search}+{n_val} questions from a pool of {len(pool)}")
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    search = [pool[i] for i in order[:n_search]]
    val = [pool[i] for i in order[n_search : n_search + n_val]]
    return search, val
