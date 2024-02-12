import math

import torch
from torch import Tensor, nn


class SinePosEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        max_len: int = 5000,
        batch_first: bool = False,
    ) -> None:
        """
        Sinusoidal positional encoding for Transformer models
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len + 1, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            pe = pe.transpose(1, 0)

        # Usually you would use register_buffer here
        # this was changed to fix a DDP bug
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if self.batch_first:
            x = x + self.pe[:, : x.size(1)]
        else:
            x = x + self.pe[: x.size(0)]

        return self.dropout(x)
