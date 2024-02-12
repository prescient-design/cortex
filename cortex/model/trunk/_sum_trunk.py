from dataclasses import dataclass

import torch
from torch import nn

from cortex.model.elemental import Expression, identity
from cortex.model.root import RootNodeOutput
from cortex.model.trunk import TrunkNode, TrunkNodeOutput


@dataclass
class PaddedTrunkOutput(TrunkNodeOutput):
    padding_mask: torch.Tensor


class SumTrunk(TrunkNode):
    """
    A trunk node which aggregates 1D sequence embeddings from one or more root nodes by summing them together.
    If the output dimension of a root node is different from the output dimension of the trunk,
    a linear projection is applied to the root features before aggregation.
    """

    def __init__(self, in_dims: list[int], out_dim: int, project_features: bool = False) -> None:
        super().__init__()
        self.encoder = Expression(identity)
        self._in_dims = in_dims
        self._out_dim = out_dim

        if project_features:
            root_projections = []
            for in_dim in self.in_dims:
                if in_dim == self.out_dim:
                    root_projections.append(identity)
                else:
                    root_projections.append(nn.Linear(in_dim, out_dim, bias=False))
            self.root_projections = nn.ModuleList(root_projections)
        # to preserve backward compatibility with trunks that do not project root features
        else:
            self.root_projections = None

    @property
    def in_dims(self) -> list[int]:
        return self._in_dims

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, *root_outputs: RootNodeOutput) -> TrunkNodeOutput:
        """
        Args:
            root_outputs: {'root_features': torch.Tensor, 'padding_mask': torch.Tensor}
        Returns:
            outputs: {'trunk_features': torch.Tensor, 'padding_mask': torch.Tensor}
        """

        if self.root_projections is None:
            features = [r_out.root_features for r_out in root_outputs]
        else:
            features = [proj(r_out.root_features) for proj, r_out in zip(self.root_projections, root_outputs)]
        agg_root_features = torch.stack(features, dim=0).sum(0)
        agg_padding_mask = torch.stack([r_out.padding_mask for r_out in root_outputs], dim=0).sum(0).clamp(0, 1)

        outputs = PaddedTrunkOutput(
            trunk_features=self.encoder(agg_root_features),
            padding_mask=agg_padding_mask,
        )
        return outputs
