from dataclasses import dataclass

import torch
from torch import nn

from cortex.model.block import TransformerBlock
from cortex.model.branch import BranchNode, BranchNodeOutput
from cortex.model.elemental import (
    Apply,
    Expression,
    MeanPooling,
    PoolingSelfAttention,
    WeightedMeanPooling,
    identity,
)
from cortex.model.trunk import PaddedTrunkOutput


@dataclass
class TransformerBranchOutput(BranchNodeOutput):
    branch_mask: torch.Tensor
    pooled_features: torch.Tensor


class TransformerBranch(BranchNode):
    """
    Branch node which transforms aggregated trunk features to task branch specific features
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 64,
        channel_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout_prob: float = 0.0,
        pooling_type: str = "mean",
        **kwargs,
    ):
        super().__init__()
        # create encoder
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.channel_dim = channel_dim
        self.num_blocks = num_blocks

        if num_blocks == 0:
            # add projection if dims don't match
            encoder_modules = [
                Expression(identity) if in_dim == out_dim else Apply(nn.Linear(in_dim, out_dim, bias=False))
            ]
        else:
            # conv layers expect inputs with shape (batch_size, input_dim, num_tokens)
            encoder_modules = []

        block_kwargs = {
            "num_heads": num_heads,
            "is_causal": is_causal,
            "dropout_p": dropout_prob,
        }

        if num_blocks == 1:
            encoder_modules.append(TransformerBlock(in_dim, out_dim, **block_kwargs))
        elif num_blocks > 1:
            encoder_modules.append(TransformerBlock(in_dim, channel_dim, **block_kwargs))
            encoder_modules.extend(
                [TransformerBlock(channel_dim, channel_dim, **block_kwargs) for _ in range(num_blocks - 2)]
            )
            encoder_modules.append(TransformerBlock(channel_dim, out_dim, **block_kwargs))

        self.encoder = nn.Sequential(*encoder_modules)
        if pooling_type == "mean":
            self.pooling_op = MeanPooling()
        elif pooling_type == "weighted_mean":
            self.pooling_op = WeightedMeanPooling(out_dim)
        elif pooling_type == "attention":
            self.pooling_op = nn.Sequential(
                Apply(nn.LayerNorm(out_dim, bias=False)),
                PoolingSelfAttention(num_heads=num_heads, embed_dim=out_dim, dropout_p=dropout_prob),
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        trunk_outputs: PaddedTrunkOutput,
    ) -> TransformerBranchOutput:
        """
        Args:
            trunk_outputs: {'trunk_features': torch.Tensor, 'padding_mask': torch.Tensor}
        Returns:
            outputs: {'branch_features': torch.Tensor, 'branch_mask': torch.Tensor, 'pooled_features': torch.Tensor}
        """
        trunk_features = trunk_outputs.trunk_features
        padding_mask = trunk_outputs.padding_mask

        branch_features, branch_mask = self.encoder((trunk_features, padding_mask.to(trunk_features)))
        pooled_features = self.pooling_op((branch_features, branch_mask))

        branch_outputs = TransformerBranchOutput(
            branch_features=branch_features.contiguous(),
            branch_mask=branch_mask,
            pooled_features=pooled_features,
        )
        return branch_outputs
