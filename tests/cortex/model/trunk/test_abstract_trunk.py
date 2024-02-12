import pytest

from cortex.model.trunk import TrunkNode


def test_trunk_node():
    with pytest.raises(TypeError):
        TrunkNode()
