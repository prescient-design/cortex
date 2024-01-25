from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BranchNodeOutput:
    branch_features: torch.Tensor


class BranchNode(ABC, nn.Module):
    """
    Receives multi-modal features from a trunk, extracts task-specific features
    and passes result to leaf nodes.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> BranchNodeOutput:
        pass
