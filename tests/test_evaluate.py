"""The naive evaluator: base equivalence, population batching, scope semantics."""

import numpy as np

from evorouter.diagnostics import router_logits_at_scored_positions, topk_margins
from evorouter.evaluate import Evaluator, pack_rows
from evorouter.genome import BiasGenome

GENOME = BiasGenome(layers=(1, 2), num_experts=8)


def make(model, questions, encode, **kw):
    return Evaluator(model, questions, encode, GENOME, max_batch_tokens=kw.pop("max_batch_tokens", 256), **kw)


def test_zero_genome_equals_base_model(model, questions, encode):
    ev = make(model, questions, encode)
    np.testing.assert_allclose(ev.loglik(GENOME.zeros()[None]), ev.loglik(None), rtol=0, atol=1e-5)


def test_population_batch_equals_one_by_one(model, questions, encode):
    rng = np.random.default_rng(0)
    thetas = rng.normal(0, 2.0, size=(3, GENOME.dim))
    ev = make(model, questions, encode)
    together = ev.loglik(thetas)
    alone = np.concatenate([ev.loglik(t[None]) for t in thetas])
    np.testing.assert_allclose(together, alone, rtol=0, atol=1e-5)
    assert not np.allclose(together[0], together[1])  # different genomes, different scores


def test_batch_budget_does_not_change_results(model, questions, encode):
    thetas = np.random.default_rng(1).normal(0, 2.0, size=(2, GENOME.dim))
    small = make(model, questions, encode, max_batch_tokens=40).loglik(thetas)
    large = make(model, questions, encode, max_batch_tokens=4096).loglik(thetas)
    np.testing.assert_allclose(small, large, rtol=0, atol=1e-5)


def test_scope_all_differs_from_scored(model, questions, encode):
    theta = np.random.default_rng(2).normal(0, 3.0, size=(1, GENOME.dim))
    scored = make(model, questions, encode, scope="scored").loglik(theta)
    everything = make(model, questions, encode, scope="all").loglik(theta)
    assert not np.allclose(scored, everything)


def test_evaluate_returns_metrics(model, questions, encode):
    s = make(model, questions, encode).evaluate(np.zeros((4, GENOME.dim)))
    assert s.fitness.shape == (4,) and np.all(s.fitness < 0)
    assert np.all((s.acc >= 0) & (s.acc <= 1))


def test_pack_rows_respects_budget():
    lengths = [10, 3, 7, 7, 2, 9]
    for batch in pack_rows(lengths, 20):
        assert len(batch) * max(lengths[i] for i in batch) <= 20 or len(batch) == 1
    assert sorted(i for b in pack_rows(lengths, 20) for i in b) == list(range(6))


def test_topk_margins(model, questions, encode):
    ev = make(model, questions, encode)
    logits = router_logits_at_scored_positions(model, ev.requests, layers=(1, 2))
    n_scored = sum(len(r.scored_positions) for r in ev.requests)
    assert logits[1].shape == (n_scored, 8)
    m = topk_margins(logits, k=2)
    assert m[2]["median"] >= 0 and m[2]["n_positions"] == n_scored
