from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from cortex.corruption._diffusion_noise_schedule import get_named_beta_schedule


class CorruptionProcess(ABC):
    """
    Base class for corruption processes, must be subclassed and
    the _corrupt method must be implemented.
    Provides noise schedule and timestep sampling, and defines
    the corruption interface.
    """

    def __init__(self, schedule: str = "cosine", max_steps: int = 1000, *args, **kwargs):
        betas = get_named_beta_schedule(schedule, max_steps)

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.max_steps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)

    def sample_timestep(self) -> int:
        return np.random.randint(1, self.max_steps + 1)

    def sample_corrupt_frac(self) -> float:
        timestep = self.sample_timestep()
        return self.timestep_to_corrupt_frac(timestep)

    def timestep_to_corrupt_frac(self, timestep: int) -> float:
        assert timestep <= self.max_steps
        if timestep == 0:
            return 0.0
        return self.sqrt_alphas_cumprod[timestep - 1]

    def __call__(
        self,
        x_start: torch.Tensor,
        timestep: Optional[int] = None,
        corrupt_frac: Optional[float] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor]:
        # can't pass both timestep and noise_frac
        assert timestep is None or corrupt_frac is None
        # infer corrupt_frac from timestep
        if timestep is not None:
            assert timestep <= self.max_steps
            corrupt_frac = self.timestep_to_corrupt_frac(timestep)
        # sample if both timestep and corrupt_frac are None
        elif corrupt_frac is None:
            corrupt_frac = self.sample_corrupt_frac()
        # return uncorrupted input if corrupt_frac is 0
        if corrupt_frac == 0:
            is_corrupted = torch.full_like(x_start, False, dtype=torch.bool)
            return x_start, is_corrupted

        x_corrupt, is_corrupted = self._corrupt(x_start, *args, corrupt_frac=corrupt_frac, **kwargs)
        # only change values where corruption_allowed is True
        if corruption_allowed is not None:
            corruption_allowed = corruption_allowed.to(x_start.device)
            x_corrupt = torch.where(corruption_allowed, x_corrupt, x_start)
            is_corrupted = torch.where(corruption_allowed, is_corrupted, False)

        return x_corrupt, is_corrupted

    @abstractmethod
    def _corrupt(self, x_start: torch.Tensor, corrupt_frac: float, *args, **kwargs):
        pass
