from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch
import torch.distributions as torch_dist

from cortex.corruption._diffusion_noise_schedule import get_named_beta_schedule


class CorruptionProcess(ABC):
    """
    Base class for corruption processes, must be subclassed and
    the _corrupt method must be implemented.
    Provides noise schedule and timestep sampling, and defines
    the corruption interface.
    """

    def __init__(
        self,
        schedule: str = "cosine",
        max_steps: int = 1000,
        t_base_dist: Optional[torch_dist.Distribution] = None,
        *args,
        **kwargs,
    ):
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

        # Set up timestep sampling distribution
        if t_base_dist is None:
            # self.t_base_dist = torch_dist.Uniform(0, 1)
            self.t_base_dist = torch_dist.Beta(1, 3)
        else:
            self.t_base_dist = t_base_dist

    def sample_timestep(self, n: int = 1):
        """Sample timestep(s) from the noise schedule.

        Args:
            n: Number of timesteps to sample. If None, returns a single int.
               If an integer, returns an array of shape (n,).

        Returns:
            Int or array of timesteps.
        """
        base_samples = self.t_base_dist.sample((n,))
        return torch.round(self.max_steps * base_samples).int()

    def sample_corrupt_frac(self, n: int = 1) -> torch.Tensor:
        """Sample corruption fraction(s).

        Args:
            n: Number of corruption fractions to sample. If None, returns a tensor with a single value.
               If an integer, returns a tensor of shape (n,).

        Returns:
            Tensor of corruption fractions.
        """
        timesteps = self.sample_timestep(n)

        if n is None:
            return torch.tensor([self.timestep_to_corrupt_frac(timesteps)])

        return torch.tensor([self.timestep_to_corrupt_frac(t) for t in timesteps])

    def timestep_to_corrupt_frac(self, timestep: int) -> float:
        assert timestep <= self.max_steps
        if timestep == 0:
            return 0.0
        return self.sqrt_alphas_cumprod[timestep - 1]

    def __call__(
        self,
        x_start: torch.Tensor,
        timestep: Optional[int] = None,
        corrupt_frac: Optional[Union[float, torch.Tensor]] = None,
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
            batch_size = x_start.shape[0]
            corrupt_frac = self.sample_corrupt_frac(n=batch_size).to(x_start.device)

        # Handle scalar and tensor corrupt_frac values consistently
        if isinstance(corrupt_frac, float):
            # If it's 0, we can early-return without corruption
            if corrupt_frac == 0:
                is_corrupted = torch.full_like(x_start, False, dtype=torch.bool)
                return x_start, is_corrupted
            # Otherwise convert to tensor matching batch dimension
            corrupt_frac = torch.full((x_start.shape[0],), corrupt_frac, device=x_start.device)

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
