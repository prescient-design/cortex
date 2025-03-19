from typing import Optional, Union

import torch

from ._abstract_corruption import CorruptionProcess


class MaskCorruptionProcess(CorruptionProcess):
    """
    Corrupt input tensor with mask values. Each tensor element is corrupted
    independently with probability `corrupt_frac`.
    """

    def __call__(
        self,
        x_start: torch.Tensor,
        mask_val: int,
        timestep: Optional[int] = None,
        corrupt_frac: Optional[float] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        return super().__call__(x_start, timestep, corrupt_frac, corruption_allowed, *args, mask_val=mask_val, **kwargs)

    def _corrupt(
        self, x_start: torch.Tensor, corrupt_frac: Union[float, torch.Tensor], mask_val: int, *args, **kwargs
    ) -> tuple[torch.Tensor]:
        # Handle per-example corrupt_frac that has a batch dimension but needs to be
        # broadcastable to the full x_start shape for element-wise operations
        if isinstance(corrupt_frac, torch.Tensor) and corrupt_frac.dim() > 0:
            # Reshape to enable broadcasting: [batch_size] -> [batch_size, 1, ...]
            corrupt_frac = corrupt_frac.view(*corrupt_frac.shape, *([1] * (x_start.dim() - corrupt_frac.dim())))

        is_corrupted = torch.rand_like(x_start, dtype=torch.float64) < corrupt_frac
        mask_tensor = torch.full_like(x_start, mask_val)
        x_corrupt = torch.where(is_corrupted, mask_tensor, x_start)
        return x_corrupt, is_corrupted
