import numpy as np
import pytest
import torch

from evorouter.genome import BiasGenome


def test_layout_is_layer_major():
    g = BiasGenome(layers=(12, 15), num_experts=4)
    theta = np.arange(8.0)
    b = g.biases(theta)
    assert g.dim == 8
    assert torch.equal(b[12], torch.tensor([0.0, 1, 2, 3]))
    assert torch.equal(b[15], torch.tensor([4.0, 5, 6, 7]))


def test_population_shape_and_scale():
    g = BiasGenome(layers=(0, 1), num_experts=3, scale=(2.0, 0.5))
    b = g.biases(np.ones((5, 6)))
    assert b[0].shape == (5, 3)
    assert torch.allclose(b[0], torch.full((5, 3), 2.0))
    assert torch.allclose(b[1], torch.full((5, 3), 0.5))


def test_validation():
    with pytest.raises(ValueError):
        BiasGenome(layers=(), num_experts=4)
    with pytest.raises(ValueError):
        BiasGenome(layers=(1, 1), num_experts=4)
    with pytest.raises(ValueError):
        BiasGenome(layers=(1,), num_experts=4).biases(np.zeros(5))
