from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from cortex.model.branch import BranchNode, BranchNodeOutput
from cortex.model.elemental import MeanPooling, WeightedMeanPooling
from cortex.model.trunk import PaddedTrunkOutput


@dataclass
class TransformerDecoderBranchOutput(BranchNodeOutput):
    """Output of TransformerDecoderBranch."""

    branch_features: torch.Tensor
    branch_mask: torch.Tensor
    pooled_features: torch.Tensor


class TransformerDecoderBranch(BranchNode):
    """
    Branch node that applies additional Transformer decoder layers with causal self-attention
    to features from the trunk.

    Example Hydra Config:
    ```yaml
    branches:
      transformer_decoder_branch:
        _target_: cortex.model.branch.TransformerDecoderBranch
        in_dim: 512  # Should match trunk output
        out_dim: 512
        num_layers: 2
        nhead: 8
        dim_feedforward: 2048  # Optional, defaults to 4 * in_dim
        dropout: 0.1
        activation: "relu"
        layer_norm_eps: 1e-5
        batch_first: True
        pooling_type: "mean"
    ```
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        pooling_type: str = "mean",
        **kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Set default dim_feedforward if not provided
        if dim_feedforward is None:
            dim_feedforward = 4 * in_dim

        # Create decoder layer and stack them
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=in_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
        )

        self.transformer_layers = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )

        # Add projection layer if dimensions don't match
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = None

        # Set up pooling operation
        if pooling_type == "mean":
            self.pooling_op = MeanPooling()
        elif pooling_type == "weighted_mean":
            self.pooling_op = WeightedMeanPooling(out_dim)
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

    def forward(
        self,
        trunk_outputs: PaddedTrunkOutput,
    ) -> TransformerDecoderBranchOutput:
        """
        Args:
            trunk_outputs: PaddedTrunkOutput containing trunk_features and padding_mask

        Returns:
            TransformerDecoderBranchOutput containing:
                branch_features: Sequence features after transformer layers
                branch_mask: Padding mask for the output sequence
                pooled_features: Pooled sequence features
        """
        features = trunk_outputs.trunk_features
        padding_mask = trunk_outputs.padding_mask

        # Convert padding_mask to tgt_key_padding_mask for transformer
        # PyTorch transformer expects True for positions to be *masked*
        tgt_key_padding_mask = ~padding_mask.bool()

        # Create causal mask to ensure autoregressive attention
        seq_len = features.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(sz=seq_len, device=features.device)

        # Apply transformer layers
        # For self-attention only, we pass features as both tgt and memory
        branch_features = self.transformer_layers(
            tgt=features,
            memory=features,  # Use features as memory for self-attention only
            tgt_mask=causal_mask,  # Apply causal masking
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=tgt_key_padding_mask,  # Same as tgt padding mask
        )

        # Apply projection if needed
        if self.projection is not None:
            branch_features = self.projection(branch_features)

        # Pool features
        pooled_features = self.pooling_op(branch_features, padding_mask)

        return TransformerDecoderBranchOutput(
            branch_features=branch_features.contiguous(),
            branch_mask=padding_mask,
            pooled_features=pooled_features,
        )
