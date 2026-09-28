"""AI2 Reasoning Challenge (ARC) from the HF hub dataset ``allenai/ai2_arc``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evorouter.tasks.base import MCQuestion

DATASET = "allenai/ai2_arc"
SUBSETS = ("ARC-Challenge", "ARC-Easy")


def from_record(rec: Mapping[str, Any]) -> MCQuestion:
    """One HF record -> MCQuestion. Labels are A-E or 1-5; answerKey uses the same scheme.

    Texts are kept verbatim (no stripping) to match lm-eval-harness ``arc_challenge`` exactly.
    """
    labels = list(rec["choices"]["label"])
    if rec["answerKey"] not in labels:
        raise ValueError(f"{rec['id']}: answerKey {rec['answerKey']!r} not in labels {labels}")
    return MCQuestion(
        id=rec["id"],
        question=rec["question"],
        choices=tuple(rec["choices"]["text"]),
        answer=labels.index(rec["answerKey"]),
    )


def load_arc(subset: str = "ARC-Challenge") -> dict[str, list[MCQuestion]]:
    """All splits of an ARC subset: {"train": [...], "validation": [...], "test": [...]}."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}")
    from datasets import load_dataset

    ds = load_dataset(DATASET, subset)
    return {split: [from_record(r) for r in ds[split]] for split in ds}
