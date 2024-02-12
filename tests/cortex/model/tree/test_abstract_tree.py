import pytest

from cortex.model.tree import NeuralTree


def test_neural_tree():
    with pytest.raises(TypeError):
        NeuralTree()
