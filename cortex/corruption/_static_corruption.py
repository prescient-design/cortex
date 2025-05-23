"""
Static corruption implementations compatible with torch.compile.

Key principles for compilation compatibility:
1. No dynamic control flow (if/else based on tensor values)
2. Fixed tensor shapes throughout computation
3. Pure tensor operations without Python loops
4. Consistent return shapes regardless of input values
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from cortex.corruption._diffusion_noise_schedule import get_named_beta_schedule


class StaticCorruptionProcess(nn.Module):
    """
    Base class for torch.compile-compatible corruption processes.

    Eliminates dynamic control flow and ensures fixed tensor shapes.
    """

    def __init__(
        self,
        schedule: str = "cosine",
        max_steps: int = 1000,
        **kwargs,
    ):
        super().__init__()

        # Precompute noise schedule as buffers (not parameters)
        betas = get_named_beta_schedule(schedule, max_steps)
        betas = torch.tensor(betas, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        # Register as buffers for device movement but not training
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        self.max_steps = max_steps

    def sample_corrupt_frac(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample corruption fractions for batch. Compilation-friendly."""
        # Use Beta(3, 1) distribution for preferential low-noise sampling
        base_samples = torch.distributions.Beta(3.0, 1.0).sample((batch_size,)).to(device)
        timesteps = torch.round(self.max_steps * base_samples).long()

        # Convert timesteps to corruption fractions
        # Handle timestep=0 case by clamping to valid range
        timesteps = torch.clamp(timesteps, 1, self.max_steps)
        corrupt_frac = self.sqrt_alphas_cumprod[timesteps - 1]

        return corrupt_frac

    def forward(
        self,
        x_start: torch.Tensor,
        corrupt_frac: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply corruption with static computation graph.

        Args:
            x_start: Input tensor [batch_size, ...]
            corrupt_frac: Corruption fractions [batch_size] or None for sampling
            corruption_allowed: Mask for allowed corruption [batch_size, ...] or None

        Returns:
            Tuple of (corrupted_tensor, corruption_mask)
        """
        batch_size = x_start.shape[0]

        # Always generate corruption fraction (no dynamic branching)
        if corrupt_frac is None:
            corrupt_frac = self.sample_corrupt_frac(batch_size, x_start.device)

        # Ensure corrupt_frac has correct shape for broadcasting
        while corrupt_frac.dim() < x_start.dim():
            corrupt_frac = corrupt_frac.unsqueeze(-1)

        # Apply corruption-specific logic
        x_corrupt, is_corrupted = self._corrupt_static(x_start, corrupt_frac, **kwargs)

        # Apply corruption_allowed mask without dynamic branching
        if corruption_allowed is not None:
            # Ensure corruption_allowed has correct shape for broadcasting
            while corruption_allowed.dim() < x_corrupt.dim():
                corruption_allowed = corruption_allowed.unsqueeze(-1)

            # Use torch.where for static computation
            x_corrupt = torch.where(corruption_allowed, x_corrupt, x_start)
            is_corrupted = torch.where(corruption_allowed, is_corrupted, torch.zeros_like(is_corrupted))

        return x_corrupt, is_corrupted

    def _corrupt_static(
        self, x_start: torch.Tensor, corrupt_frac: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subclass-specific corruption logic. Must be compilation-friendly."""
        raise NotImplementedError


class StaticMaskCorruption(StaticCorruptionProcess):
    """
    Mask corruption compatible with torch.compile.

    Eliminates dynamic control flow from original MaskCorruptionProcess.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        x_start: torch.Tensor,
        mask_val: int,
        corrupt_frac: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with mask value."""
        return super().forward(
            x_start, corrupt_frac=corrupt_frac, corruption_allowed=corruption_allowed, mask_val=mask_val, **kwargs
        )

    def _corrupt_static(
        self, x_start: torch.Tensor, corrupt_frac: torch.Tensor, mask_val: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Static mask corruption without dynamic shapes."""

        # Generate corruption mask with fixed computation
        corruption_probs = torch.rand_like(x_start, dtype=torch.float32)
        is_corrupted = corruption_probs < corrupt_frac

        # Create mask tensor and apply corruption
        mask_tensor = torch.full_like(x_start, mask_val)
        x_corrupt = torch.where(is_corrupted, mask_tensor, x_start)

        return x_corrupt, is_corrupted


class StaticGaussianCorruption(StaticCorruptionProcess):
    """
    Gaussian noise corruption compatible with torch.compile.

    Eliminates dynamic operations from original GaussianCorruptionProcess.
    """

    def __init__(self, noise_variance: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_variance = noise_variance

    def _corrupt_static(
        self, x_start: torch.Tensor, corrupt_frac: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Static Gaussian corruption without dynamic operations."""

        # Compute noise scale statically
        noise_scale = corrupt_frac * math.sqrt(self.noise_variance)

        # Apply noise with fixed computation
        noise = torch.randn_like(x_start.float())
        x_corrupt = (1.0 - corrupt_frac) * x_start.float() + noise_scale * noise

        # All elements are considered corrupted for Gaussian noise
        is_corrupted = torch.ones_like(x_start, dtype=torch.bool)

        return x_corrupt, is_corrupted


class StaticCorruptionFactory:
    """Factory for creating compilation-compatible corruption processes."""

    @staticmethod
    def create_mask_corruption(**kwargs) -> StaticMaskCorruption:
        """Create static mask corruption process."""
        return StaticMaskCorruption(**kwargs)

    @staticmethod
    def create_gaussian_corruption(noise_variance: float = 10.0, **kwargs) -> StaticGaussianCorruption:
        """Create static Gaussian corruption process."""
        return StaticGaussianCorruption(noise_variance=noise_variance, **kwargs)
