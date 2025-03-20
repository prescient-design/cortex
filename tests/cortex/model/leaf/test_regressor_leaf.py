import torch

from cortex.model.branch import Conv1dBranchOutput
from cortex.model.leaf import RegressorLeaf, RegressorLeafOutput
from cortex.model.root import RootNodeOutput


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


def test_regressor_leaf_per_element_alpha():
    """Test RegressorLeaf with per-element alpha (label smoothing)."""
    in_dim = 2
    out_dim = 3
    batch_size = 5
    max_seq_len = 7

    # Create leaf node that will use corrupt_frac for label smoothing
    leaf_node = RegressorLeaf(in_dim=in_dim, out_dim=out_dim, branch_key="test", label_smoothing="corrupt_frac")

    # Create inputs
    branch_features = torch.rand(batch_size, max_seq_len, in_dim)
    branch_output = Conv1dBranchOutput(
        branch_features=branch_features,
        branch_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
        pooled_features=branch_features.mean(-2),
    )

    # Create output
    leaf_output = leaf_node(branch_output)

    # Create targets (random values)
    targets = torch.rand(batch_size, out_dim)

    # Test case 1: scalar corrupt_frac
    root_output1 = RootNodeOutput(root_features=torch.rand(batch_size, max_seq_len, in_dim), corrupt_frac=0.1)

    # The loss should run without errors
    loss1 = leaf_node.loss(leaf_output, root_output1, targets)
    assert torch.is_tensor(loss1)
    assert loss1.ndim == 0  # Loss should be a scalar tensor

    # Test case 2: per-element corrupt_frac
    per_element_corrupt_frac = torch.rand(batch_size)  # Different value for each example
    root_output2 = RootNodeOutput(
        root_features=torch.rand(batch_size, max_seq_len, in_dim), corrupt_frac=per_element_corrupt_frac
    )

    # The loss should run without errors
    loss2 = leaf_node.loss(leaf_output, root_output2, targets)
    assert torch.is_tensor(loss2)
    assert loss2.ndim == 0  # Loss should be a scalar tensor
