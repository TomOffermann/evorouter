import numpy as np
import pytest

from evorouter.scoring import PROMPT, build_requests, mc_scores
from evorouter.tasks.arc import from_record
from evorouter.tasks.base import MCQuestion, search_val_split


def test_requests_split_context_and_continuation(questions, encode):
    reqs = build_requests(questions, encode)
    assert len(reqs) == 4 + 3 + 5
    r = reqs[1]  # q1, "wind"
    ctx = PROMPT.format(question="Which is renewable?")
    assert r.n_ctx == len(ctx.encode())
    assert bytes(r.tokens[r.n_ctx :]).decode() == " wind"
    assert list(r.scored_positions) == list(range(r.n_ctx - 1, len(r.tokens) - 1))


def test_metrics_on_hand_made_logliks():
    qs = [MCQuestion("a", "?", ("x", "yyyy"), 1), MCQuestion("b", "?", ("zz", "w", "vvv"), 0)]
    reqs = build_requests(qs, lambda s: list(s.encode()))
    # q a: raw LL prefers "x" (-1 > -2) but per-char prefers "yyyy" (-0.5 > -1)
    # q b: "zz" best under both
    ll = np.array([[-1.0, -2.0, -0.2, -3.0, -3.0]])
    s = mc_scores(ll, reqs, qs)
    assert s.acc[0] == pytest.approx(0.5)
    assert s.acc_norm[0] == pytest.approx(1.0)
    norm_a = np.array([-1.0, -0.5])
    norm_b = np.array([-0.1, -3.0, -1.0])
    expected = (norm_a[1] - np.log(np.exp(norm_a).sum()) + norm_b[0] - np.log(np.exp(norm_b).sum())) / 2
    assert s.fitness[0] == pytest.approx(expected)


def test_population_axis():
    qs = [MCQuestion("a", "?", ("x", "y"), 0)]
    reqs = build_requests(qs, lambda s: list(s.encode()))
    s = mc_scores(np.array([[-1.0, -2.0], [-2.0, -1.0]]), reqs, qs)
    assert s.acc.tolist() == [1.0, 0.0]
    assert s.fitness[0] > s.fitness[1]


def test_arc_record_labels():
    rec = {
        "id": "x",
        "question": "Q?",
        "choices": {"text": ["a", "b", "c"], "label": ["1", "2", "3"]},
        "answerKey": "3",
    }
    assert from_record(rec).answer == 2
    rec = {
        "id": "y",
        "question": "Q?",
        "choices": {"text": ["a", "b"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    assert from_record(rec).answer == 1


def test_search_val_split_is_disjoint_and_seeded(questions):
    pool = questions * 10
    pool = [MCQuestion(f"{i}", q.question, q.choices, q.answer) for i, q in enumerate(pool)]
    s1, v1 = search_val_split(pool, seed=0, n_search=5, n_val=10)
    s2, _ = search_val_split(pool, seed=0, n_search=5, n_val=10)
    assert [q.id for q in s1] == [q.id for q in s2]
    assert not {q.id for q in s1} & {q.id for q in v1}
