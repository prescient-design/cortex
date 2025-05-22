import torch
from torch import nn


class MeanPooling(nn.Module):
    """
    Average pooling over the sequence dimension excluding padding token positions.
    """

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, padding_mask = inputs
        weights = torch.where(padding_mask.bool(), 0.0, float("-inf"))
        weights = weights.softmax(dim=-1).to(x)
        pooled_x = (x * weights[..., None]).sum(-2)
        return pooled_x


class WeightedMeanPooling(nn.Module):
    """
    Weighted average pooling over the sequence dimension excluding padding token positions.
    Weights are learned by a linear layer. Breaks fused Adam optimizer.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, in_dim)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, padding_mask = inputs
        weights = self.encoder(x)
        weights = torch.where(padding_mask.bool().unsqueeze(-1), weights, float("-inf"))
        weights = weights.softmax(dim=-2).to(x)
        pooled_x = (x * weights).sum(-2)
        return pooled_x
