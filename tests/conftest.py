import pytest
import torch

from evorouter.models import byte_encode, tiny_olmoe
from evorouter.tasks.base import MCQuestion


@pytest.fixture(scope="session")
def model():
    return tiny_olmoe()


@pytest.fixture
def ids():
    torch.manual_seed(1)
    return torch.randint(1, 256, (2, 12))


@pytest.fixture(scope="session")
def questions():
    return [
        MCQuestion("q1", "Which is renewable?", ("coal", "wind", "oil", "gas"), 1),
        MCQuestion("q2", "What do plants need?", ("light", "sand", "rock"), 0),
        MCQuestion("q3", "Largest planet?", ("Mars", "Venus", "Jupiter", "Earth", "Pluto"), 2),
    ]


@pytest.fixture(scope="session")
def encode():
    return byte_encode
