from abc import ABC
from dataclasses import dataclass

from torch import nn


@dataclass
class LeafNodeOutput:
    pass


class LeafNode(ABC, nn.Module):
    """
    Receives task-specific features and returns task outputs
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError
