import pytest

from cortex.model.branch import BranchNode


def test_branch_node():
    with pytest.raises(TypeError):
        BranchNode()
