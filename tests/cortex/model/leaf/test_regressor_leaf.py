import torch

from cortex.model.branch import Conv1dBranchOutput
from cortex.model.leaf import RegressorLeaf, RegressorLeafOutput


def test_regressor_leaf():
    in_dim = 2
    out_dim = 3
    batch_size = 5
    max_seq_len = 7

    leaf_node = RegressorLeaf(in_dim=in_dim, out_dim=out_dim, branch_key="test")

    branch_features = torch.rand(batch_size, max_seq_len, in_dim)
    branch_output = Conv1dBranchOutput(
        branch_features=branch_features,
        branch_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
        pooled_features=branch_features.mean(-2),
    )
    leaf_output = leaf_node(branch_output)
    assert isinstance(leaf_output, RegressorLeafOutput)
    loc = leaf_output.loc
    scale = leaf_output.scale
    assert torch.is_tensor(loc)
    assert loc.size() == torch.Size((batch_size, out_dim))
    assert torch.is_tensor(scale)
    assert scale.size() == torch.Size((batch_size, out_dim))
    assert torch.all(scale > 0)
