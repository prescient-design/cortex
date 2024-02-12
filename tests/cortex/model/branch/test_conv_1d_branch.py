import torch

from cortex.model.branch import Conv1dBranch, Conv1dBranchOutput
from cortex.model.trunk import PaddedTrunkOutput


def test_conv_1d_branch():
    in_dim = 2
    out_dim = 3
    embed_dim = 5
    num_blocks = 7
    kernel_size = 11
    max_seq_len = 13
    batch_size = 17
    dropout_prob = 0.125
    layernorm = True

    branch_node = Conv1dBranch(
        in_dim=in_dim,
        out_dim=out_dim,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        kernel_size=kernel_size,
        dropout_prob=dropout_prob,
        layernorm=layernorm,
    )

    trunk_output = PaddedTrunkOutput(
        trunk_features=torch.rand(batch_size, max_seq_len, in_dim),
        padding_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
    )
    branch_output = branch_node(trunk_output)
    assert isinstance(branch_output, Conv1dBranchOutput)
    branch_features = branch_output.branch_features
    branch_mask = branch_output.branch_mask
    pooled_features = branch_output.pooled_features

    assert torch.is_tensor(branch_features)
    assert torch.is_tensor(branch_mask)
    assert torch.is_tensor(pooled_features)

    assert branch_features.size() == torch.Size((batch_size, max_seq_len, out_dim))
    assert branch_mask.size() == torch.Size((batch_size, max_seq_len))
    assert pooled_features.size() == torch.Size((batch_size, out_dim))
