import torch

from cortex.model.branch import Conv1dBranchOutput
from cortex.model.leaf import ClassifierLeaf, ClassifierLeafOutput
from cortex.model.root import RootNodeOutput


def test_classifier_leaf():
    in_dim = 2
    num_classes = 3
    batch_size = 5
    max_seq_len = 7

    leaf_node = ClassifierLeaf(in_dim=in_dim, num_classes=num_classes, branch_key="test")

    branch_features = torch.rand(batch_size, max_seq_len, in_dim)
    branch_output = Conv1dBranchOutput(
        branch_features=branch_features,
        branch_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
        pooled_features=branch_features.mean(-2),
    )
    leaf_output = leaf_node(branch_output)
    assert isinstance(leaf_output, ClassifierLeafOutput)
    logits = leaf_output.logits
    assert torch.is_tensor(logits)
    assert logits.size() == torch.Size((batch_size, num_classes))


def test_classifier_leaf_per_element_alpha():
    """Test ClassifierLeaf with per-element alpha (label smoothing)."""
    in_dim = 2
    num_classes = 3
    batch_size = 5
    max_seq_len = 7

    # Create leaf node that will use corrupt_frac for label smoothing
    leaf_node = ClassifierLeaf(
        in_dim=in_dim, num_classes=num_classes, branch_key="test", label_smoothing="corrupt_frac"
    )

    # Create inputs
    branch_features = torch.rand(batch_size, max_seq_len, in_dim)
    branch_output = Conv1dBranchOutput(
        branch_features=branch_features,
        branch_mask=torch.ones(batch_size, max_seq_len, dtype=torch.float),
        pooled_features=branch_features.mean(-2),
    )

    # Create output with per-element corrupt_frac
    leaf_output = leaf_node(branch_output)

    # Create targets (random class indices)
    targets = torch.randint(0, num_classes, (batch_size,))

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
