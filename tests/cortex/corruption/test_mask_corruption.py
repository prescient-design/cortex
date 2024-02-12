import torch

from cortex.corruption import MaskCorruptionProcess


def test_mask_corruption():
    """
    Test the mask corruption process.
    """
    corruption_process = MaskCorruptionProcess()

    # 0th timestep should return uncorrupted input
    x_start = torch.arange(1, 128)
    x_corrupt, is_corrupted = corruption_process(x_start, timestep=0, corruption_allowed=None, mask_val=0)
    assert torch.allclose(x_start, x_corrupt)
    assert not torch.any(is_corrupted)

    # random timestep should return corrupted input
    x_corrupt, is_corrupted = corruption_process(x_start, corrupt_frac=0.5, mask_val=0)
    assert x_start.size() == x_corrupt.size()
    assert torch.any(is_corrupted)
    assert torch.all(torch.masked_select(x_corrupt, is_corrupted) == 0)

    # input should be unchanged where corruption_allowed is False
    corruption_allowed = torch.rand_like(x_start, dtype=torch.float64) < 0.5
    timestep = corruption_process.max_steps // 2
    x_corrupt, is_corrupted = corruption_process(
        x_start, timestep=timestep, corruption_allowed=corruption_allowed, mask_val=0
    )
    assert torch.any(torch.masked_select(is_corrupted, corruption_allowed))
    assert not torch.any(torch.masked_select(is_corrupted, ~corruption_allowed))
    assert not torch.any(torch.masked_select(x_corrupt, ~corruption_allowed) == 0)
