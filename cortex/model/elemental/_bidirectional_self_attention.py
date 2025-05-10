from torch import Tensor, nn


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, num_heads: int = 4, embed_dim: int = 32, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("num_heads must evenly divide embed_dim")

        self.c_attn = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, padding_mask = inputs
        batch_size, seq_len, embed_dim = x.shape
        queries, keys, values = self.c_attn(x).split(embed_dim, dim=-1)

        queries = queries.view(-1, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        keys = keys.view(-1, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)
        values = values.view(-1, seq_len, self.num_heads, self.head_dim).transpose(-2, -3)

        attn_mask = padding_mask[..., None, :, None]
        attn_mask = attn_mask.expand(-1, -1, -1, seq_len).contiguous()

        res = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        res = res.transpose(-2, -3).contiguous().flatten(start_dim=-2)
        return self.dropout(res), padding_mask
