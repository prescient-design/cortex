import torch

from cortex.model.branch import TransformerEncoderBranch, TransformerEncoderBranchOutput
from cortex.model.trunk import PaddedTrunkOutput


def test_conv_1d_branch():
    in_dim = 12
    out_dim = 12
    embed_dim = 12
    channel_dim = 12
    num_blocks = 7
    num_heads = 3
    max_seq_len = 13
    batch_size = 17
    dropout_prob = 0.125
    layernorm = True

    branch_node = TransformerEncoderBranch(
        in_dim=in_dim,
        out_dim=out_dim,
        embed_dim=embed_dim,
        channel_dim=channel_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_prob=dropout_prob,
        layernorm=layernorm,
    )

    trunk_output = PaddedTrunkOutput(
        trunk_features=torch.rand(batch_size, max_seq_len, in_dim),
        padding_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
    )
    branch_output = branch_node(trunk_output)
    assert isinstance(branch_output, TransformerEncoderBranchOutput)
    branch_features = branch_output.branch_features
    branch_mask = branch_output.branch_mask
    pooled_features = branch_output.pooled_features

    assert torch.is_tensor(branch_features)
    assert torch.is_tensor(branch_mask)
    assert torch.is_tensor(pooled_features)

    assert branch_features.size() == torch.Size((batch_size, max_seq_len, out_dim))
    assert branch_mask.size() == torch.Size((batch_size, max_seq_len))
    assert pooled_features.size() == torch.Size((batch_size, out_dim))
