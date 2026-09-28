"""Exact, hard-routed evaluation of genome populations (the naive reference evaluator).

Every (genome, request) pair is a row; rows are packed into padded batches under a token budget
and run through the full model, each row with its own genome's biases. This is slow but obviously
correct, and it is the ground truth that any faster (cached) evaluator must reproduce.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from typing import Literal

import numpy as np
import torch
from torch import nn

from evorouter.genome import BiasGenome
from evorouter.routing import Mode, RoutingIntervention, apply_routing
from evorouter.scoring import Encode, MCScores, Request, build_requests, mc_scores
from evorouter.tasks.base import MCQuestion

Scope = Literal["scored", "all"]


def pack_rows(lengths: Sequence[int], max_batch_tokens: int) -> list[list[int]]:
    """Greedy packing of rows (longest first) into batches with rows * max_len <= max_batch_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches: list[list[int]] = []
    for i in order:
        if batches and (len(batches[-1]) + 1) * lengths[batches[-1][0]] <= max_batch_tokens:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


class Evaluator:
    """Scores a population of genomes on a fixed set of multiple-choice questions."""

    def __init__(
        self,
        model: nn.Module,
        questions: Sequence[MCQuestion],
        encode: Encode,
        genome: BiasGenome | None = None,
        *,
        mode: Mode = "selection",
        scope: Scope = "scored",
        tau: float = 1.0,
        max_batch_tokens: int = 16384,
        pad_id: int = 0,
    ) -> None:
        if scope not in ("scored", "all"):
            raise ValueError(f"unknown scope {scope!r}")
        self.model = model
        self.questions = list(questions)
        self.requests: list[Request] = build_requests(self.questions, encode)
        self.genome = genome
        self.mode = mode
        self.scope = scope
        self.tau = tau
        self.max_batch_tokens = max_batch_tokens
        self.pad_id = pad_id
        self.device = next(model.parameters()).device

    def subset(self, question_indices: Sequence[int]) -> Evaluator:
        """Evaluator on a subset of the questions (e.g. a mini-batch), reusing the tokenized requests."""
        idx = list(question_indices)
        remap = {q: i for i, q in enumerate(idx)}
        if len(remap) != len(idx):
            raise ValueError("duplicate question indices")
        sub = copy.copy(self)
        sub.questions = [self.questions[q] for q in idx]
        sub.requests = sorted(
            (replace(r, question=remap[r.question]) for r in self.requests if r.question in remap),
            key=lambda r: (r.question, r.choice),
        )
        return sub

    @property
    def num_tokens(self) -> int:
        """Tokens in one pass over all requests (one genome)."""
        return sum(len(r.tokens) for r in self.requests)

    @torch.no_grad()
    def loglik(self, thetas: np.ndarray | None = None) -> np.ndarray:
        """Continuation log-likelihoods [N, len(requests)]; ``thetas`` [N, dim] or None for the base model."""
        if thetas is None:
            n_members, biases = 1, None
        else:
            if self.genome is None:
                raise ValueError("thetas given but the evaluator has no genome")
            thetas = np.atleast_2d(thetas)
            n_members, biases = thetas.shape[0], self.genome.biases(thetas, self.device)

        rows = [(m, r) for m in range(n_members) for r in range(len(self.requests))]
        lengths = [len(self.requests[r].tokens) for _, r in rows]
        out = np.zeros((n_members, len(self.requests)))

        for batch in pack_rows(lengths, self.max_batch_tokens):
            members = [rows[i][0] for i in batch]
            reqs = [self.requests[rows[i][1]] for i in batch]
            max_len = max(len(r.tokens) for r in reqs)
            ids = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            pmask = torch.zeros((len(batch), max_len), dtype=torch.bool)
            b_idx, p_idx, targets = [], [], []
            for bi, r in enumerate(reqs):
                ids[bi, : len(r.tokens)] = torch.tensor(r.tokens)
                attn[bi, : len(r.tokens)] = 1
                positions = list(r.scored_positions)
                if self.scope == "scored":
                    pmask[bi, positions] = True
                else:
                    pmask[bi, : len(r.tokens)] = True
                b_idx += [bi] * len(positions)
                p_idx += positions
                targets += list(r.tokens[r.n_ctx :])

            ids, attn, pmask = ids.to(self.device), attn.to(self.device), pmask.to(self.device)
            if biases is None:
                ctx = nullcontext()
            else:
                member_idx = torch.tensor(members, device=self.device)
                row_biases = {layer: b[member_idx] for layer, b in biases.items()}
                ctx = apply_routing(self.model, RoutingIntervention(row_biases, self.mode, pmask))
            with ctx:
                logits = self.model(input_ids=ids, attention_mask=attn, use_cache=False).logits

            b_t = torch.tensor(b_idx, device=self.device)
            tok_lp = (
                logits[b_t, torch.tensor(p_idx, device=self.device)]
                .float()
                .log_softmax(-1)
                .gather(-1, torch.tensor(targets, device=self.device)[:, None])
                .squeeze(-1)
            )
            sums = torch.zeros(len(batch), dtype=torch.float64, device=self.device)
            sums.index_add_(0, b_t, tok_lp.double())
            for bi, i in enumerate(batch):
                m, r = rows[i]
                out[m, r] = sums[bi].item()
        return out

    def evaluate(self, thetas: np.ndarray | None = None) -> MCScores:
        return mc_scores(self.loglik(thetas), self.requests, self.questions, self.tau)
