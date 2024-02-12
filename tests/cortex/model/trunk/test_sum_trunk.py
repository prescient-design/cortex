import torch

from cortex.model.root import Conv1dRootOutput
from cortex.model.trunk import PaddedTrunkOutput, SumTrunk


def test_sum_trunk():
    num_roots = 3
    in_dim = 4
    max_seq_len = 5
    batch_size = 7

    trunk_node = SumTrunk(in_dims=[in_dim] * num_roots, out_dim=in_dim)
    root_outputs = [
        Conv1dRootOutput(
            root_features=torch.rand(batch_size, max_seq_len, in_dim),
            padding_mask=torch.ones(batch_size, max_seq_len),
        )
        for _ in range(num_roots)
    ]

    trunk_outputs = trunk_node(*root_outputs)
    assert isinstance(trunk_outputs, PaddedTrunkOutput)
    trunk_features = trunk_outputs.trunk_features
    padding_mask = trunk_outputs.padding_mask

    assert torch.is_tensor(trunk_features)
    assert torch.is_tensor(padding_mask)

    assert trunk_features.size() == torch.Size((batch_size, max_seq_len, in_dim))
    assert padding_mask.size() == torch.Size((batch_size, max_seq_len))
