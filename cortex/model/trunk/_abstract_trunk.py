from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TrunkNodeOutput:
    """
    Dataclass for output of TrunkNode.
    """

    trunk_features: torch.Tensor


class TrunkNode(ABC, nn.Module):
    """
    Mixes mode-specific features from different roots, extracts multi-modal features
    and passes result to branch nodes.
    Cannot be instantiated directly, must be subclassed.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractproperty
    def in_dims(self) -> list[int]:
        """
        Returns:
            in_dims: list of input dimensions
        """
        pass

    @abstractproperty
    def out_dim(self) -> int:
        """
        Returns:
            out_dim: output dimension of aggregated features
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> TrunkNodeOutput:
        pass
