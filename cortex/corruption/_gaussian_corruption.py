import math

import torch

from ._abstract_corruption import CorruptionProcess


class GaussianCorruptionProcess(CorruptionProcess):
    """
    Corrupt input tensor with additive Gaussian noise with
    variance `noise_variance`. Each tensor element is corrupted
    independently with probability `corrupt_frac`.
    """

    def __init__(self, noise_variance: float = 10.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_variance = noise_variance

    def _corrupt(self, x_start: torch.Tensor, corrupt_frac: float, *args, **kwargs) -> tuple[torch.Tensor]:
        noise_scale = corrupt_frac * math.sqrt(self.noise_variance)
        x_corrupt = (1.0 - corrupt_frac) * x_start + noise_scale * torch.randn_like(x_start)
        is_corrupted = torch.ones_like(x_start, dtype=torch.bool)
        return x_corrupt, is_corrupted
