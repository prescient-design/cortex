import math
from typing import Union

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

    def _corrupt(
        self, x_start: torch.Tensor, corrupt_frac: Union[float, torch.Tensor], *args, **kwargs
    ) -> tuple[torch.Tensor]:
        # Handle per-example corrupt_frac that has a batch dimension but needs to be
        # broadcastable to the full x_start shape for element-wise operations
        if isinstance(corrupt_frac, torch.Tensor) and corrupt_frac.dim() > 0:
            # Reshape to enable broadcasting: [batch_size] -> [batch_size, 1, ...]
            corrupt_frac = corrupt_frac.view(*corrupt_frac.shape, *([1] * (x_start.dim() - corrupt_frac.dim())))

        noise_scale = corrupt_frac * math.sqrt(self.noise_variance)
        x_corrupt = (1.0 - corrupt_frac) * x_start + noise_scale * torch.randn_like(x_start)
        is_corrupted = torch.ones_like(x_start, dtype=torch.bool)
        return x_corrupt, is_corrupted
