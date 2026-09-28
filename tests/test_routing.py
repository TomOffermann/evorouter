"""The routing intervention: identity at zero bias, selection-only semantics, masking, batching."""

import pytest
import torch
import torch.nn.functional as F

from evorouter.routing import RoutingIntervention, apply_routing, last_layers, moe_blocks

E = 8  # experts in the tiny model


def logits_of(model, ids, intervention=None):
    if intervention is None:
        return model(ids).logits
    with apply_routing(model, intervention):
        return model(ids).logits


def test_zero_bias_is_bit_identical_to_base(model, ids):
    base = logits_of(model, ids)
    zeros = {layer: torch.zeros(E) for layer in range(3)}
    for mode in ("selection", "logit"):
        assert torch.equal(logits_of(model, ids, RoutingIntervention(zeros, mode)), base)


def test_forward_restored_after_context(model, ids):
    base = logits_of(model, ids)
    logits_of(model, ids, RoutingIntervention({2: torch.full((E,), 5.0)}))
    assert "forward" not in vars(moe_blocks(model)[2])
    assert torch.equal(logits_of(model, ids), base)


def test_selection_mode_changes_selection_but_keeps_gates(model, ids):
    bias = torch.zeros(E)
    bias[3] = 100.0  # force expert 3 into every top-k
    iv = RoutingIntervention({2: bias}, "selection", record=True)
    logits_of(model, ids, iv)
    rec = iv.records[2]
    assert (rec.selected == 3).any(-1).all()
    unbiased = F.softmax(rec.logits, dim=-1).gather(1, rec.selected)
    torch.testing.assert_close(rec.weights, unbiased)


def test_logit_mode_moves_the_gates(model, ids):
    bias = torch.zeros(E)
    bias[3] = 100.0
    iv = RoutingIntervention({2: bias}, "logit", record=True)
    logits_of(model, ids, iv)
    rec = iv.records[2]
    w3 = rec.weights[rec.selected == 3]
    assert torch.all(w3 > 0.99)  # expert 3 now takes almost all gate mass


def test_bias_changes_output(model, ids):
    bias = torch.zeros(E)
    bias[3] = 100.0
    assert not torch.allclose(logits_of(model, ids, RoutingIntervention({2: bias})), logits_of(model, ids))


def test_position_mask_leaves_earlier_positions_untouched(model, ids):
    bias = torch.zeros(E)
    bias[3] = 100.0
    mask = torch.zeros(ids.shape, dtype=torch.bool)
    mask[:, -1] = True  # bias only the last position
    out = logits_of(model, ids, RoutingIntervention({1: bias, 2: bias}, position_mask=mask))
    base = logits_of(model, ids)
    assert torch.equal(out[:, :-1], base[:, :-1])  # causal: earlier positions cannot see the change
    assert not torch.allclose(out[:, -1], base[:, -1])


def test_per_row_biases_match_separate_runs(model, ids):
    torch.manual_seed(0)
    biases = torch.randn(2, E)  # one bias per batch row
    batched = logits_of(model, ids, RoutingIntervention({2: biases}))
    for row in range(2):
        single = logits_of(model, ids[row : row + 1], RoutingIntervention({2: biases[row]}))
        torch.testing.assert_close(batched[row : row + 1], single)


def test_last_layers_and_validation(model):
    assert last_layers(model, 2) == (1, 2)
    with pytest.raises(ValueError):
        last_layers(model, 4)
    with pytest.raises(ValueError), apply_routing(model, RoutingIntervention({7: torch.zeros(E)})):
        pass
