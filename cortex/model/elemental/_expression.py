from typing import Callable

from torch import Tensor, nn


class Expression(nn.Module):
    """
    `nn.Module` wrapper for arbitrary function (useful for `nn.Sequential`)
    """

    def __init__(self, func: Callable) -> None:
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        return self.func(x)
