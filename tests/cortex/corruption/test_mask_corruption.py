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


def test_sample_corrupt_frac_with_n():
    """Test the sample_corrupt_frac method with n parameter."""
    corruption_process = MaskCorruptionProcess()

    # Test with n=None (default behavior)
    scalar_corrupt_frac = corruption_process.sample_corrupt_frac()
    assert isinstance(scalar_corrupt_frac, torch.Tensor)
    assert scalar_corrupt_frac.ndim == 1
    assert scalar_corrupt_frac.shape[0] == 1
    assert 0.0 <= scalar_corrupt_frac.item() <= 1.0

    # Test with n=1 (should return tensor with 1 value)
    single_corrupt_frac = corruption_process.sample_corrupt_frac(n=1)
    assert isinstance(single_corrupt_frac, torch.Tensor)
    assert single_corrupt_frac.ndim == 1
    assert single_corrupt_frac.shape[0] == 1
    assert 0.0 <= single_corrupt_frac.item() <= 1.0

    # Test with n=5 (should return tensor with 5 values)
    batch_size = 5
    batch_corrupt_frac = corruption_process.sample_corrupt_frac(n=batch_size)
    assert isinstance(batch_corrupt_frac, torch.Tensor)
    assert batch_corrupt_frac.ndim == 1
    assert batch_corrupt_frac.shape[0] == batch_size
    assert torch.all(batch_corrupt_frac >= 0.0)
    assert torch.all(batch_corrupt_frac <= 1.0)

    # Verify that different elements can have different values
    # (This is probabilistic, but with high probability values should differ)
    more_samples = corruption_process.sample_corrupt_frac(n=100)
    assert more_samples.shape[0] == 100
    assert more_samples.unique().shape[0] > 1, "Sample corrupt_frac values should vary"
