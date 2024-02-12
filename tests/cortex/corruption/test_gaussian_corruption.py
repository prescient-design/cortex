import torch

from cortex.corruption import GaussianCorruptionProcess


def test_gaussian_corruption():
    """
    Test the Gaussian corruption process.
    """
    corruption_process = GaussianCorruptionProcess()

    # 0th timestep should return uncorrupted input
    x_start = torch.arange(128, dtype=torch.float64)
    x_corrupt, is_corrupted = corruption_process(x_start, timestep=0, corruption_allowed=None)
    assert torch.allclose(x_start, x_corrupt)
    assert not torch.any(is_corrupted)

    # random timestep should return corrupted input
    x_corrupt, is_corrupted = corruption_process(x_start)
    assert x_start.size() == x_corrupt.size()
    assert torch.any(is_corrupted)

    # input should be unchanged where corruption_allowed is False
    corruption_allowed = torch.rand_like(x_start) < 0.5
    timestep = corruption_process.max_steps // 2
    x_corrupt, is_corrupted = corruption_process(x_start, timestep=timestep, corruption_allowed=corruption_allowed)
    assert torch.any(torch.masked_select(is_corrupted, corruption_allowed))
    assert not torch.any(torch.masked_select(is_corrupted, ~corruption_allowed))
