from typing import Optional

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
        self, x_start: torch.Tensor, corrupt_frac: float, mask_val: int, *args, **kwargs
    ) -> tuple[torch.Tensor]:
        is_corrupted = torch.rand_like(x_start, dtype=torch.float64) < corrupt_frac
        mask_tensor = torch.full_like(x_start, mask_val)
        x_corrupt = torch.where(is_corrupted, mask_tensor, x_start)
        return x_corrupt, is_corrupted
