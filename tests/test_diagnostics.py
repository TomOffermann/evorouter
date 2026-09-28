"""Diagnostics helpers: selection change rate, temperature fit, stats, sharding."""

import numpy as np
import pytest
import torch

from evorouter.context import shard
from evorouter.diagnostics import expert_usage, router_logits_at_scored_positions, selection_change_rate
from evorouter.scoring import build_requests, fit_temperature, mc_scores
from evorouter.stats import rankdata, spearman


def test_selection_change_rate_zero_and_forced(model, questions, encode):
    reqs = build_requests(questions, encode)
    zero = selection_change_rate(model, reqs, {2: torch.zeros(8)})
    assert zero == {2: 0.0}
    force = torch.zeros(8)
    force[5] = 50.0
    logits = router_logits_at_scored_positions(model, reqs, layers=(2,))
    already = expert_usage(logits, k=2)[2][5]  # share of positions where expert 5 is selected anyway
    rate = selection_change_rate(model, reqs, {2: force})[2]
    assert rate == pytest.approx(1.0 - already)


def test_scope_all_changes_more_positions_downstream(model, questions, encode):
    reqs = build_requests(questions, encode)
    force = torch.zeros(8)
    force[5] = 50.0
    rates = selection_change_rate(model, reqs, {1: force, 2: torch.zeros(8)}, scope="all")
    assert rates[1] > 0 and rates[2] > 0  # layer-2 selections change only through layer 1


def test_fit_temperature_prefers_calibration():
    from evorouter.tasks.base import MCQuestion

    qs = [MCQuestion(str(i), "?", ("a", "b"), 0) for i in range(20)]
    reqs = build_requests(qs, lambda s: list(s.encode()))
    # correct option is better by 0.05 per char in 15 questions, worse in 5: weakly informative
    ll = np.array([[-1.0, -1.05] * 15 + [-1.05, -1.0] * 5])
    tau = fit_temperature(ll, reqs, qs)
    assert tau < 1.0  # sharper than tau=1 because the per-char scores are compressed
    assert mc_scores(ll, reqs, qs, tau).fitness[0] > mc_scores(ll, reqs, qs, 1.0).fitness[0]


def test_spearman_and_ranks():
    assert spearman(np.arange(5), np.arange(5) ** 2) == pytest.approx(1.0)
    assert spearman(np.arange(5), -np.arange(5)) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.ones(4), np.arange(4)))
    assert rankdata(np.array([3.0, 1.0, 3.0])).tolist() == [1.5, 0.0, 1.5]


def test_shard_partitions():
    items = list(range(10))
    parts = [shard(items, i, 4) for i in range(4)]
    assert sorted(x for p in parts for x in p) == items
    with pytest.raises(ValueError):
        shard(items, 4, 4)
