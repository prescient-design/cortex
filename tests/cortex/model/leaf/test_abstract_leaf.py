import pytest

from cortex.model.leaf import LeafNode


def test_leaf_node():
    leaf_node = LeafNode()
    with pytest.raises(NotImplementedError):
        leaf_node()
