import torch

from cortex.model.elemental import MLP


def test_mlp():
    in_channels = 32
    module = MLP(in_channels)

    x = torch.randn(2, 3, in_channels)
    res = module(x)

    assert res.shape == x.shape
