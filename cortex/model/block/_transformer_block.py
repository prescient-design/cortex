from torch import Tensor, nn

from cortex.model.elemental import MLP, BidirectionalSelfAttention, CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        bias: bool = False,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_channels, bias=bias)

        if is_causal:
            self.attn = CausalSelfAttention(num_heads=num_heads, embed_dim=in_channels, dropout_p=dropout_p, bias=bias)
        else:
            self.attn = BidirectionalSelfAttention(
                num_heads=num_heads, embed_dim=in_channels, dropout_p=dropout_p, bias=bias
            )

        self.ln_2 = nn.LayerNorm(in_channels, bias=bias)
        self.mlp = MLP(in_channels, out_channels, bias=bias, dropout_p=dropout_p)

        if not in_channels == out_channels:
            self.proj = nn.Linear(in_channels, out_channels, bias=bias)
        else:
            self.proj = None

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        resid, padding_mask = inputs
        x, padding_mask = self.attn((self.ln_1(resid), padding_mask))
        x = resid + x

        if self.proj is not None:
            resid = self.proj(resid)

        x = resid + self.mlp(self.ln_2(x))

        return x, padding_mask
