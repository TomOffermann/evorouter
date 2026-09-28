"""lm-eval-style cloze scoring of multiple-choice questions.

Each (question, choice) pair becomes one token sequence: context ``Question: <q>\\nAnswer:``
followed by the continuation `` <choice>``. Tokenization follows lm-eval-harness: encode the
context and the full string, take the continuation tokens as ``full[len(context):]``.

The model reads the continuation at the *scored positions* n_ctx-1 .. len-2 (the position before
each continuation token). These are the positions a genome acts on under ``scope="scored"``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from evorouter.tasks.base import MCQuestion

PROMPT = "Question: {question}\nAnswer:"
Encode = Callable[[str], list[int]]


@dataclass(frozen=True)
class Request:
    question: int  # index into the question list
    choice: int
    tokens: tuple[int, ...]  # context + continuation
    n_ctx: int  # number of context tokens

    @property
    def n_cont(self) -> int:
        return len(self.tokens) - self.n_ctx

    @property
    def scored_positions(self) -> range:
        """Positions whose next-token prediction is a continuation token."""
        return range(self.n_ctx - 1, len(self.tokens) - 1)


def build_requests(questions: Sequence[MCQuestion], encode: Encode) -> list[Request]:
    requests = []
    for qi, q in enumerate(questions):
        context = PROMPT.format(question=q.question)
        ctx_ids = list(encode(context))
        for ci, choice in enumerate(q.choices):
            full = list(encode(context + " " + choice))
            cont = full[len(ctx_ids) :]
            if not cont:
                raise ValueError(f"{q.id} choice {ci}: empty continuation after tokenization")
            requests.append(Request(qi, ci, tuple(ctx_ids + cont), len(ctx_ids)))
    return requests


@dataclass(frozen=True)
class MCScores:
    """Per-member metrics over a question set. Arrays have shape [N] (population size)."""

    fitness: np.ndarray  # mean log q(correct), q = softmax over options of length-normalized LL / tau
    acc: np.ndarray  # argmax of summed LL
    acc_norm: np.ndarray  # argmax of LL / len(choice), as lm-eval acc_norm

    def member(self, i: int) -> dict[str, float]:
        return {
            "fitness": float(self.fitness[i]),
            "acc": float(self.acc[i]),
            "acc_norm": float(self.acc_norm[i]),
        }


def mc_scores(
    loglik: np.ndarray, requests: Sequence[Request], questions: Sequence[MCQuestion], tau: float = 1.0
) -> MCScores:
    """Aggregate continuation log-likelihoods ``loglik`` [N, len(requests)] into per-member metrics."""
    loglik = np.atleast_2d(np.asarray(loglik, dtype=np.float64))
    n_members = loglik.shape[0]
    n_q, a_max = len(questions), max(len(q.choices) for q in questions)
    ll = np.full((n_members, n_q, a_max), -np.inf)
    length = np.ones((n_q, a_max))
    for j, r in enumerate(requests):
        ll[:, r.question, r.choice] = loglik[:, j]
        length[r.question, r.choice] = max(len(questions[r.question].choices[r.choice]), 1)
    answers = np.array([q.answer for q in questions])
    rows = np.arange(n_q)

    acc = (ll.argmax(-1) == answers).mean(-1)
    norm = ll / length  # -inf stays -inf for missing options
    acc_norm = (norm.argmax(-1) == answers).mean(-1)

    z = norm / tau
    z_max = z.max(-1, keepdims=True)
    log_q = z - z_max - np.log(np.exp(z - z_max).sum(-1, keepdims=True))
    fitness = log_q[:, rows, answers].mean(-1)
    return MCScores(fitness=fitness, acc=acc, acc_norm=acc_norm)
