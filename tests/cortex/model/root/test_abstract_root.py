import pytest

from cortex.model.root import RootNode


def test_root_node():
    with pytest.raises(TypeError):
        RootNode()
