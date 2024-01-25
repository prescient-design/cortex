from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RootNodeOutput:
    root_features: torch.Tensor


class RootNode(ABC, nn.Module):
    """
    Input node, receives e.g. sequences, point-clouds, graphs, images, or tabular features.
    Extracts mode-specific features and passes result to a trunk node.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> RootNodeOutput:
        pass
