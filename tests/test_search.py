"""Resumable search runs with per-generation mini-batches, on the tiny model."""

import json

import numpy as np
import pytest

from evorouter.evaluate import Evaluator
from evorouter.genome import BiasGenome
from evorouter.search import SearchConfig, SearchRun
from evorouter.tasks.base import MCQuestion

GENOME = BiasGenome(layers=(1, 2), num_experts=8)


@pytest.fixture(scope="module")
def pool(questions):
    return [
        MCQuestion(f"p{i}", q.question + f" #{i}", q.choices, q.answer) for i, q in enumerate(questions * 3)
    ]


def make_run(tmp_path, model, pool, encode, **cfg_kw):
    cfg = SearchConfig(**{"popsize": 4, "val_every": 2, "max_generations": 4, "batch_size": 4, **cfg_kw})
    train = Evaluator(model, pool, encode, GENOME, max_batch_tokens=512)
    val = Evaluator(model, pool[:3], encode, GENOME, max_batch_tokens=512)
    return SearchRun(tmp_path, cfg, train_eval=train, val_eval=val, test_eval=val)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("optimizer", ["cma", "sep-cma", "random"])
def test_run_to_completion(tmp_path, model, pool, encode, optimizer):
    run = make_run(tmp_path, model, pool, encode, optimizer=optimizer)
    assert run.run(time_budget_s=600, log=lambda _: None)
    metrics = lines(tmp_path / "metrics.jsonl")
    assert [m["generation"] for m in metrics] == [1, 2, 3, 4]
    assert metrics[-1]["evaluations"] == 16
    assert all(m["batch_questions"] == 4 for m in metrics)
    assert metrics[0]["mean_minus_base_fitness"] == pytest.approx(0.0, abs=1e-6)  # mean starts at base
    assert [v["generation"] for v in lines(tmp_path / "val.jsonl")] == [0, 2, 4]
    final = json.loads((tmp_path / "final.json").read_text())
    assert len(final["theta"]) == GENOME.dim
    assert final["val"]["fitness"] >= lines(tmp_path / "val.jsonl")[0]["fitness"]  # never worse than base


def test_batches_are_fresh_and_deterministic(tmp_path, model, pool, encode):
    run = make_run(tmp_path, model, pool, encode)
    ids = [tuple(q.id for q in run.batch(g).questions) for g in range(6)]
    assert len(set(ids)) > 1  # different questions in different generations
    assert ids[3] == tuple(q.id for q in run.batch(3).questions)  # same generation -> same batch
    full = make_run(tmp_path / "full", model, pool, encode, batch_size=0)
    assert len(full.batch(0).questions) == len(pool)


def test_subset_matches_fresh_evaluator(model, pool, encode):
    thetas = np.random.default_rng(0).normal(0, 2.0, size=(2, GENOME.dim))
    full = Evaluator(model, pool, encode, GENOME, max_batch_tokens=512)
    idx = [1, 4, 7]
    via_subset = full.subset(idx).loglik(thetas)
    fresh = Evaluator(model, [pool[i] for i in idx], encode, GENOME, max_batch_tokens=512).loglik(thetas)
    np.testing.assert_allclose(via_subset, fresh, rtol=0, atol=1e-5)


def test_resume_continues_where_it_stopped(tmp_path, model, pool, encode):
    run = make_run(tmp_path, model, pool, encode, max_generations=2)
    run.run(time_budget_s=600, log=lambda _: None)
    mean_after_2 = np.asarray(run.state["optimizer"].mean).copy()

    resumed = make_run(tmp_path, model, pool, encode, max_generations=4)
    assert resumed.state["generation"] == 2
    np.testing.assert_array_equal(np.asarray(resumed.state["optimizer"].mean), mean_after_2)
    resumed.run(time_budget_s=600, log=lambda _: None)
    assert [m["generation"] for m in lines(tmp_path / "metrics.jsonl")] == [1, 2, 3, 4]


def test_time_budget_pauses_and_saves(tmp_path, model, pool, encode):
    run = make_run(tmp_path, model, pool, encode)
    assert not run.run(time_budget_s=0.0, log=lambda _: None)
    assert (tmp_path / "state.pkl").exists()
    assert not (tmp_path / "final.json").exists()


def test_batch_size_larger_than_pool_is_rejected(tmp_path, model, pool, encode):
    with pytest.raises(ValueError):
        make_run(tmp_path, model, pool, encode, batch_size=len(pool) + 1)
