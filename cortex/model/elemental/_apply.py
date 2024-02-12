from typing import Callable, Iterable

from torch import nn


class Apply(nn.Module):
    """
    `nn.Module` which applies a function to a specific dimension of the input
    """

    def __init__(self, module: Callable, dim: int = 0) -> None:
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x: Iterable) -> Iterable:
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return xs
