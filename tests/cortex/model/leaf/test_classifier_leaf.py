import torch

from cortex.model.branch import Conv1dBranchOutput
from cortex.model.leaf import ClassifierLeaf, ClassifierLeafOutput


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
