"""Resumable search runs on the tiny model."""

import json

import numpy as np
import pytest

from evorouter.evaluate import Evaluator
from evorouter.genome import BiasGenome
from evorouter.search import SearchConfig, SearchRun

GENOME = BiasGenome(layers=(1, 2), num_experts=8)


def make_run(tmp_path, model, questions, encode, **cfg_kw):
    cfg = SearchConfig(**{"popsize": 4, "val_every": 2, "max_generations": 4, **cfg_kw})
    ev = Evaluator(model, questions, encode, GENOME, max_batch_tokens=512)
    return SearchRun(tmp_path, cfg, search_eval=ev, val_eval=ev, test_eval=ev)


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("optimizer", ["cma", "sep-cma", "random"])
def test_run_to_completion(tmp_path, model, questions, encode, optimizer):
    run = make_run(tmp_path, model, questions, encode, optimizer=optimizer)
    assert run.run(time_budget_s=600, log=lambda _: None)
    metrics = lines(tmp_path / "metrics.jsonl")
    assert [m["generation"] for m in metrics] == [1, 2, 3, 4]
    assert metrics[-1]["evaluations"] == 16
    assert [v["generation"] for v in lines(tmp_path / "val.jsonl")] == [0, 2, 4]
    final = json.loads((tmp_path / "final.json").read_text())
    assert len(final["theta"]) == GENOME.dim
    assert final["val"]["fitness"] >= lines(tmp_path / "val.jsonl")[0]["fitness"]  # never worse than base


def test_resume_continues_where_it_stopped(tmp_path, model, questions, encode):
    run = make_run(tmp_path, model, questions, encode, max_generations=2)
    run.run(time_budget_s=600, log=lambda _: None)
    mean_after_2 = np.asarray(run.state["optimizer"].mean).copy()

    resumed = make_run(tmp_path, model, questions, encode, max_generations=4)
    assert resumed.state["generation"] == 2
    np.testing.assert_array_equal(np.asarray(resumed.state["optimizer"].mean), mean_after_2)
    resumed.run(time_budget_s=600, log=lambda _: None)
    assert [m["generation"] for m in lines(tmp_path / "metrics.jsonl")] == [1, 2, 3, 4]


def test_time_budget_pauses_and_saves(tmp_path, model, questions, encode):
    run = make_run(tmp_path, model, questions, encode)
    assert not run.run(time_budget_s=0.0, log=lambda _: None)
    assert (tmp_path / "state.pkl").exists()
    assert not (tmp_path / "final.json").exists()
